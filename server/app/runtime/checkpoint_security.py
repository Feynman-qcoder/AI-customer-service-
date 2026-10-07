"""Checkpoint filesystem, topology and revalidation security contracts.

Location validation does not open SQLite or hold a process lock. The storage
provider revalidates the attestation and acquires the process lock immediately
before opening storage.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_ATTESTATION_SEAL = object()
_REMOTE_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "davfs",
        "glusterfs",
        "nfs",
        "nfs4",
        "smb3",
        "smbfs",
        "sshfs",
    }
)
_WORKER_ENV_NAMES = (
    "WEB_CONCURRENCY",
    "UVICORN_WORKERS",
    "GUNICORN_WORKERS",
    "API_WORKERS",
    "SERVER_WORKERS",
    "WORKER_COUNT",
)
_WORKER_ARG_ENV_NAMES = ("UVICORN_CMD_ARGS", "GUNICORN_CMD_ARGS")
_WORKER_ARGUMENT = re.compile(
    r"(?:^|\s)(?:--workers|-w)(?:=|\s+)(\d+)(?:\s|$)"
)


class CheckpointSettingsView(Protocol):
    checkpoint_required: bool
    checkpoint_db_path: str
    checkpoint_worker_count: int


class CheckpointStorageError(RuntimeError):
    """Stable content-free checkpoint security diagnostic."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"checkpoint storage rejected: {reason}")


@dataclass(frozen=True, slots=True)
class CanonicalCheckpointPath:
    configured_absolute: Path
    canonical_path: Path
    traversed_link_or_reparse: bool


@dataclass(frozen=True, slots=True)
class FilesystemClassification:
    filesystem_type: str
    verified: bool
    remote: bool


@dataclass(frozen=True, slots=True)
class FilesystemObjectSecurity:
    identity: str
    kind: str
    owner_matches: bool
    permissions_verified: bool
    permissions_secure: bool
    read_only: bool
    security_digest: str


@dataclass(frozen=True, slots=True)
class WorkerTopologyEvidence:
    configured_workers: int
    checked_sources: tuple[str, ...]
    authoritative: bool
    process_lock_required: bool


class ValidatedCheckpointLocation:
    """Sealed attestation issued only by CheckpointFilesystemSecurity."""

    __slots__ = (
        "_canonical_path",
        "_filesystem_type",
        "_identity_digest",
        "_seal",
    )

    _canonical_path: Path
    _filesystem_type: str
    _identity_digest: str
    _seal: object

    def __init__(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise TypeError("validated checkpoint locations are security-issued")

    @property
    def canonical_path(self) -> Path:
        return self._canonical_path

    @property
    def filesystem_type(self) -> str:
        return self._filesystem_type

    @property
    def identity_digest(self) -> str:
        return self._identity_digest

    def _is_genuine(self) -> bool:
        return self._seal is _ATTESTATION_SEAL

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("validated checkpoint locations are immutable")


def _issue_validated_checkpoint_location(
    *,
    canonical_path: Path,
    filesystem_type: str,
    identity_digest: str,
) -> ValidatedCheckpointLocation:
    """Module-private issuer; ordinary callers cannot use the public type to mint."""

    location = object.__new__(ValidatedCheckpointLocation)
    object.__setattr__(location, "_canonical_path", canonical_path)
    object.__setattr__(location, "_filesystem_type", filesystem_type)
    object.__setattr__(location, "_identity_digest", identity_digest)
    object.__setattr__(location, "_seal", _ATTESTATION_SEAL)
    return location


class CheckpointProcessLockPort(Protocol):
    """5.2/5.3 lifecycle contract; no production implementation in 5.1."""

    def acquire(self, location: ValidatedCheckpointLocation) -> None: ...

    def release(self) -> None: ...


class CheckpointFilesystemSecurityPort(Protocol):
    def validate(
        self,
        configured_path: str,
        *,
        repository_root: Path,
    ) -> ValidatedCheckpointLocation: ...

    def revalidate(
        self,
        configured_path: str,
        attestation: ValidatedCheckpointLocation,
        *,
        repository_root: Path,
    ) -> ValidatedCheckpointLocation: ...


class CheckpointPlatformAdapter(Protocol):
    platform_name: str

    def canonicalize(
        self, configured_path: str, repository_root: Path
    ) -> CanonicalCheckpointPath: ...

    def classify_filesystem(self, path: Path) -> FilesystemClassification: ...

    def repository_path_is_ignored(
        self, path: Path, repository_root: Path
    ) -> bool: ...

    def ensure_parent(self, path: Path) -> None: ...

    def inspect_directory(self, path: Path) -> FilesystemObjectSecurity: ...

    def inspect_target(self, path: Path) -> FilesystemObjectSecurity | None: ...

    def probe_writable(self, path: Path) -> None: ...


class PlatformCheckpointFilesystemAdapter:
    """Real Windows/POSIX adapter behind the narrow security port."""

    platform_name = "windows" if os.name == "nt" else "posix"

    def canonicalize(
        self, configured_path: str, repository_root: Path
    ) -> CanonicalCheckpointPath:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        lexical = Path(os.path.abspath(os.path.normpath(str(candidate))))
        traversed = self._contains_link_or_reparse(lexical)
        raw = str(lexical)
        if raw.startswith("\\\\") or raw.startswith("//"):
            canonical = lexical
        else:
            try:
                canonical = lexical.resolve(strict=False)
            except OSError:
                raise CheckpointStorageError("CANONICAL_PATH_UNVERIFIABLE") from None
        return CanonicalCheckpointPath(
            configured_absolute=lexical,
            canonical_path=canonical,
            traversed_link_or_reparse=traversed,
        )

    def classify_filesystem(self, path: Path) -> FilesystemClassification:
        if self.platform_name == "windows":
            return self._classify_windows_filesystem(path)
        return self._classify_posix_filesystem(path)

    def repository_path_is_ignored(
        self, path: Path, repository_root: Path
    ) -> bool:
        try:
            resolved_root = repository_root.resolve(strict=False)
            relative_target = path.relative_to(resolved_root)
            relative_parent = path.parent.relative_to(resolved_root)
        except ValueError:
            return True
        try:
            tracked = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    relative_target.as_posix(),
                ],
                cwd=repository_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            if tracked.returncode == 0:
                return False
            if tracked.returncode != 1:
                raise CheckpointStorageError("IGNORE_STATUS_UNVERIFIABLE")
            result = subprocess.run(
                [
                    "git",
                    "check-ignore",
                    "--quiet",
                    "--no-index",
                    "--",
                    (relative_parent / ".checkpoint-directory-probe").as_posix(),
                ],
                cwd=repository_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            raise CheckpointStorageError("IGNORE_STATUS_UNVERIFIABLE") from None
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise CheckpointStorageError("IGNORE_STATUS_UNVERIFIABLE")

    def ensure_parent(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise CheckpointStorageError("UNWRITABLE_PATH") from None

    def inspect_directory(self, path: Path) -> FilesystemObjectSecurity:
        return self._inspect_object(path, expected_directory=True)

    def inspect_target(self, path: Path) -> FilesystemObjectSecurity | None:
        try:
            exists = path.exists() or path.is_symlink()
        except OSError:
            raise CheckpointStorageError("FILESYSTEM_UNVERIFIABLE") from None
        if not exists:
            return None
        return self._inspect_object(path, expected_directory=False)

    def probe_writable(self, path: Path) -> None:
        try:
            with tempfile.NamedTemporaryFile(dir=path, prefix=".checkpoint-probe-"):
                pass
        except OSError:
            raise CheckpointStorageError("UNWRITABLE_PATH") from None

    def _contains_link_or_reparse(self, path: Path) -> bool:
        chain = [path, *path.parents]
        for component in reversed(chain):
            try:
                if not component.exists() and not component.is_symlink():
                    continue
                metadata = os.lstat(component)
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                return True
            attributes = getattr(metadata, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if attributes & reparse:
                return True
        return False

    @staticmethod
    def _classify_windows_filesystem(path: Path) -> FilesystemClassification:
        raw = str(path)
        if raw.startswith("\\\\") or raw.startswith("//"):
            return FilesystemClassification("unc-smb", True, True)
        anchor = path.anchor
        if not anchor:
            return FilesystemClassification("unknown", False, False)
        try:
            drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(str(anchor)))
        except (AttributeError, OSError, ValueError):
            return FilesystemClassification("unknown", False, False)
        names = {
            0: "unknown",
            1: "no-root",
            2: "removable",
            3: "fixed",
            4: "mapped-remote",
            5: "cdrom",
            6: "ramdisk",
        }
        name = names.get(drive_type, "unknown")
        return FilesystemClassification(
            name,
            drive_type not in {0, 1},
            drive_type == 4,
        )

    @staticmethod
    def _classify_posix_filesystem(path: Path) -> FilesystemClassification:
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.is_file():
            return FilesystemClassification("unknown", False, False)
        try:
            lines = mountinfo.read_text(encoding="utf-8").splitlines()
        except OSError:
            return FilesystemClassification("unknown", False, False)
        resolved = str(path.resolve(strict=False))
        best: tuple[int, str] | None = None
        for line in lines:
            left, separator, right = line.partition(" - ")
            if not separator:
                continue
            left_fields = left.split()
            right_fields = right.split()
            if len(left_fields) < 5 or not right_fields:
                continue
            mount_point = (
                left_fields[4]
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\")
            )
            if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
                if best is None or len(mount_point) > best[0]:
                    best = (len(mount_point), right_fields[0].lower())
        if best is None:
            return FilesystemClassification("unknown", False, False)
        filesystem_type = best[1]
        remote = filesystem_type in _REMOTE_FILESYSTEMS or filesystem_type.startswith(
            "fuse."
        )
        return FilesystemClassification(filesystem_type, True, remote)

    def _inspect_object(
        self, path: Path, *, expected_directory: bool
    ) -> FilesystemObjectSecurity:
        try:
            metadata = os.lstat(path)
        except OSError:
            raise CheckpointStorageError("FILESYSTEM_UNVERIFIABLE") from None
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
            raise CheckpointStorageError("LINK_OR_REPARSE_POINT")
        kind = (
            "directory"
            if stat.S_ISDIR(metadata.st_mode)
            else "regular-file"
            if stat.S_ISREG(metadata.st_mode)
            else "other"
        )
        identity = f"{metadata.st_dev}:{metadata.st_ino}"
        if self.platform_name == "windows":
            verified, secure, owner_matches, acl_digest = self._windows_acl(path)
            read_only_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
            read_only = bool(attributes & read_only_flag)
        else:
            mode = stat.S_IMODE(metadata.st_mode)
            get_effective_uid = getattr(os, "geteuid", None)
            if get_effective_uid is None:
                raise CheckpointStorageError("PERMISSIONS_UNVERIFIABLE")
            owner_matches = metadata.st_uid == int(get_effective_uid())
            if expected_directory:
                secure = (mode & 0o700) == 0o700 and (mode & 0o077) == 0
                read_only = (mode & 0o200) == 0
            else:
                secure = (mode & 0o600) == 0o600 and (mode & 0o077) == 0
                read_only = (mode & 0o200) == 0
            verified = True
            acl_digest = hashlib.sha256(f"{mode:o}:{metadata.st_uid}".encode()).hexdigest()
        return FilesystemObjectSecurity(
            identity=identity,
            kind=kind,
            owner_matches=owner_matches,
            permissions_verified=verified,
            permissions_secure=secure,
            read_only=read_only,
            security_digest=acl_digest,
        )

    @staticmethod
    def _windows_acl(path: Path) -> tuple[bool, bool, bool, str]:
        script = r"""
$ErrorActionPreference = 'Stop'
$securityModule = Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1'
Import-Module -Force -Name $securityModule
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$acl = Get-Acl -LiteralPath $env:CHECKPOINT_SECURITY_PATH
$sidType = [System.Security.Principal.SecurityIdentifier]
$owner = $acl.Owner
try { $owner = ([System.Security.Principal.NTAccount]$acl.Owner).Translate($sidType).Value } catch {}
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$access = @($acl.Access | ForEach-Object {
  $sid = $_.IdentityReference.Value
  try { $sid = $_.IdentityReference.Translate($sidType).Value } catch {}
  [PSCustomObject]@{
    sid = $sid
    allow = ($_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Allow)
    rights = [Int64]$_.FileSystemRights
  }
})
[PSCustomObject]@{ owner = $owner; current = $current; access = $access } |
  ConvertTo-Json -Compress -Depth 5
        """
        environment = dict(os.environ)
        environment["CHECKPOINT_SECURITY_PATH"] = str(path)
        windows_root = Path(environment.get("SystemRoot", r"C:\Windows"))
        shell = (
            windows_root
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )
        if not shell.is_file():
            return False, False, False, "unverified"
        try:
            result = subprocess.run(
                [
                    str(shell),
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=8,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False, False, False, "unverified"
            document = json.loads(result.stdout)
            owner = str(document["owner"])
            current = str(document["current"])
            raw_access = document.get("access", [])
            entries = raw_access if isinstance(raw_access, list) else [raw_access]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            return False, False, False, "unverified"
        allowed_sids = {current, "S-1-5-18", "S-1-5-32-544"}
        secure = True
        normalized: list[tuple[str, bool, int]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return False, False, False, "unverified"
            sid = str(entry.get("sid", ""))
            allow = bool(entry.get("allow", False))
            rights = int(entry.get("rights", 0))
            normalized.append((sid, allow, rights))
            if allow and rights != 0 and sid not in allowed_sids:
                secure = False
        digest_payload = json.dumps(
            {
                "owner": owner,
                "current": current,
                "access": sorted(normalized),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        owner_matches = owner in allowed_sids
        readable_writable = os.access(path, os.R_OK | os.W_OK)
        if path.is_dir():
            readable_writable = readable_writable and os.access(path, os.X_OK)
        return (
            True,
            secure and owner_matches and readable_writable,
            owner_matches,
            hashlib.sha256(digest_payload).hexdigest(),
        )


class CheckpointFilesystemSecurity:
    """High-level validator and sealed-attestation issuer."""

    def __init__(self, *, adapter: CheckpointPlatformAdapter | None = None) -> None:
        self._adapter = adapter or PlatformCheckpointFilesystemAdapter()

    def validate(
        self,
        configured_path: str,
        *,
        repository_root: Path,
    ) -> ValidatedCheckpointLocation:
        canonical, classification, parent, target = self._collect(
            configured_path,
            repository_root=repository_root,
            create_parent=True,
            probe=True,
        )
        return _issue_validated_checkpoint_location(
            canonical_path=canonical,
            filesystem_type=classification.filesystem_type,
            identity_digest=self._identity_digest(
                canonical, classification, parent, target
            ),
        )

    def revalidate(
        self,
        configured_path: str,
        attestation: ValidatedCheckpointLocation,
        *,
        repository_root: Path,
    ) -> ValidatedCheckpointLocation:
        if not attestation._is_genuine():
            raise CheckpointStorageError("ATTESTATION_INVALID")
        canonical, classification, parent, target = self._collect(
            configured_path,
            repository_root=repository_root,
            create_parent=False,
            probe=False,
        )
        digest = self._identity_digest(canonical, classification, parent, target)
        if (
            canonical != attestation.canonical_path
            or classification.filesystem_type != attestation.filesystem_type
            or digest != attestation.identity_digest
        ):
            raise CheckpointStorageError("ATTESTATION_MISMATCH")
        return _issue_validated_checkpoint_location(
            canonical_path=canonical,
            filesystem_type=classification.filesystem_type,
            identity_digest=digest,
        )

    def _collect(
        self,
        configured_path: str,
        *,
        repository_root: Path,
        create_parent: bool,
        probe: bool,
    ) -> tuple[
        Path,
        FilesystemClassification,
        FilesystemObjectSecurity,
        FilesystemObjectSecurity | None,
    ]:
        canonical = self._adapter.canonicalize(configured_path, repository_root)
        if canonical.traversed_link_or_reparse:
            raise CheckpointStorageError("LINK_OR_REPARSE_POINT")
        classification = self._adapter.classify_filesystem(canonical.canonical_path)
        self._validate_filesystem(classification)

        repository = repository_root.resolve(strict=False)
        try:
            in_repository = canonical.canonical_path.is_relative_to(repository)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            in_repository = str(canonical.canonical_path).startswith(str(repository))
        if in_repository and not self._adapter.repository_path_is_ignored(
            canonical.canonical_path, repository
        ):
            raise CheckpointStorageError("REPOSITORY_TRACKED_PATH")

        if create_parent:
            self._adapter.ensure_parent(canonical.canonical_path.parent)
            repeated = self._adapter.canonicalize(configured_path, repository_root)
            if (
                repeated.traversed_link_or_reparse
                or repeated.canonical_path != canonical.canonical_path
            ):
                raise CheckpointStorageError("CANONICAL_PATH_CHANGED")
            classification = self._adapter.classify_filesystem(
                repeated.canonical_path
            )
            self._validate_filesystem(classification)

        parent = self._adapter.inspect_directory(canonical.canonical_path.parent)
        target = self._adapter.inspect_target(canonical.canonical_path)
        self._validate_object(parent, expected_kind="directory", target=False)
        if target is not None:
            self._validate_object(
                target,
                expected_kind="regular-file",
                target=True,
            )
        if probe:
            self._adapter.probe_writable(canonical.canonical_path.parent)
        return canonical.canonical_path, classification, parent, target

    @staticmethod
    def _validate_filesystem(classification: FilesystemClassification) -> None:
        if not classification.verified:
            raise CheckpointStorageError("FILESYSTEM_UNVERIFIABLE")
        if classification.remote:
            raise CheckpointStorageError("REMOTE_FILESYSTEM")

    @staticmethod
    def _validate_object(
        value: FilesystemObjectSecurity,
        *,
        expected_kind: str,
        target: bool,
    ) -> None:
        if value.kind != expected_kind:
            reason = "TARGET_NOT_REGULAR_FILE" if target else "PARENT_NOT_DIRECTORY"
            raise CheckpointStorageError(reason)
        if not value.owner_matches or not value.permissions_verified:
            raise CheckpointStorageError("PERMISSIONS_UNVERIFIABLE")
        if value.read_only:
            reason = "TARGET_READ_ONLY" if target else "UNWRITABLE_PATH"
            raise CheckpointStorageError(reason)
        if not value.permissions_secure:
            raise CheckpointStorageError("PERMISSIONS_TOO_PERMISSIVE")

    @staticmethod
    def _identity_digest(
        canonical: Path,
        classification: FilesystemClassification,
        parent: FilesystemObjectSecurity,
        target: FilesystemObjectSecurity | None,
    ) -> str:
        payload = {
            "canonical": str(canonical),
            "filesystem_type": classification.filesystem_type,
            "parent": {
                "identity": parent.identity,
                "security": parent.security_digest,
            },
            "target": None
            if target is None
            else {
                "identity": target.identity,
                "security": target.security_digest,
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_worker_topology(
    settings: CheckpointSettingsView,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorkerTopologyEvidence:
    """Reject known multi-worker sources without claiming process authority."""

    environment = os.environ if environ is None else environ
    checked = ["CHECKPOINT_WORKER_COUNT"]
    if settings.checkpoint_worker_count != 1:
        raise CheckpointStorageError("WORKER_TOPOLOGY_CONFLICT")
    for name in _WORKER_ENV_NAMES:
        raw = environment.get(name)
        if raw is None or not raw.strip():
            continue
        checked.append(name)
        try:
            workers = int(raw)
        except ValueError:
            raise CheckpointStorageError("WORKER_TOPOLOGY_UNVERIFIABLE") from None
        if workers != 1:
            raise CheckpointStorageError("WORKER_TOPOLOGY_CONFLICT")
    for name in _WORKER_ARG_ENV_NAMES:
        raw = environment.get(name)
        if raw is None or not raw.strip():
            continue
        checked.append(name)
        match = _WORKER_ARGUMENT.search(raw)
        if match is not None and int(match.group(1)) != 1:
            raise CheckpointStorageError("WORKER_TOPOLOGY_CONFLICT")
    return WorkerTopologyEvidence(
        configured_workers=1,
        checked_sources=tuple(sorted(checked)),
        authoritative=False,
        process_lock_required=True,
    )


def validate_checkpoint_storage(
    settings: CheckpointSettingsView,
    *,
    security: CheckpointFilesystemSecurityPort | None = None,
    environ: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
) -> ValidatedCheckpointLocation | None:
    """Validate and attest without opening SQLite or writing a target file."""

    if not settings.checkpoint_required:
        return None
    validate_checkpoint_worker_topology(settings, environ=environ)
    root = repository_root or Path(__file__).resolve().parents[3]
    implementation = security or CheckpointFilesystemSecurity()
    return implementation.validate(
        settings.checkpoint_db_path,
        repository_root=root,
    )


def revalidate_checkpoint_location(
    settings: CheckpointSettingsView,
    attestation: ValidatedCheckpointLocation,
    *,
    security: CheckpointFilesystemSecurityPort | None = None,
    environ: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
) -> ValidatedCheckpointLocation:
    """Mandatory 5.2 pre-open check; mismatch fails before open/write."""

    if not settings.checkpoint_required:
        raise CheckpointStorageError("ATTESTATION_NOT_REQUIRED")
    validate_checkpoint_worker_topology(settings, environ=environ)
    root = repository_root or Path(__file__).resolve().parents[3]
    implementation = security or CheckpointFilesystemSecurity()
    return implementation.revalidate(
        settings.checkpoint_db_path,
        attestation,
        repository_root=root,
    )
