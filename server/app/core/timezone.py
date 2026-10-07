from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _asia_shanghai_timezone() -> tzinfo:
    try:
        return ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), name="Asia/Shanghai")


ASIA_SHANGHAI: Final[tzinfo] = _asia_shanghai_timezone()


def utc_now_naive() -> datetime:
    """Return the current UTC instant in the database's naive storage form."""
    return datetime.now(UTC).replace(tzinfo=None)


def to_asia_shanghai(value: datetime) -> datetime:
    """Convert a stored UTC-naive or aware datetime for API presentation."""
    source = value.replace(tzinfo=UTC) if value.tzinfo is None else value
    return source.astimezone(ASIA_SHANGHAI)


def format_asia_shanghai(value: datetime | None) -> str:
    if value is None:
        return "暂未同步"
    return f"{to_asia_shanghai(value):%Y-%m-%d %H:%M}（北京时间）"
