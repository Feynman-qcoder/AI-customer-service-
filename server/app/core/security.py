import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.db.models import UserAccount
from app.db.session import get_session


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: int
    username: str
    name: str
    role: str


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user: UserAccount) -> str:
    return create_access_token_claims(user.id, user.username, user.display_name, user.role)


def create_access_token_claims(user_id: int, username: str, display_name: str, role: str) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "name": display_name,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    header_part = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_part = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_part}.{payload_part}"
    signature = hmac.new(settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = f"{header_part}.{payload_part}"
        expected = _b64url(hmac.new(settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature_part):
            raise UnauthorizedError()
        payload_raw = json.loads(_b64url_decode(payload_part))
        if not isinstance(payload_raw, dict):
            raise UnauthorizedError()
        payload = cast(dict[str, Any], payload_raw)
        if int(payload["exp"]) < int(time.time()):
            raise UnauthorizedError()
        return payload
    except UnauthorizedError:
        raise
    except Exception as exc:
        raise UnauthorizedError() from exc


def verify_password(raw_password: str, password_hash: str) -> bool:
    import bcrypt

    return bool(bcrypt.checkpw(raw_password.encode("utf-8"), password_hash.encode("utf-8")))


def hash_password(raw_password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> AuthenticatedUser:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise UnauthorizedError()
    payload = decode_access_token(authorization.removeprefix("Bearer ").strip())
    user = await session.get(UserAccount, int(payload["sub"]))
    if user is None or user.status != "ACTIVE":
        raise UnauthorizedError()
    return AuthenticatedUser(user_id=user.id, username=user.username, name=user.display_name, role=user.role)


def require_admin(user: AuthenticatedUser = Depends(current_user)) -> AuthenticatedUser:
    if user.role != "ADMIN":
        raise ForbiddenError()
    return user
