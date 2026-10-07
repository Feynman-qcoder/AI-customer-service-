from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @staticmethod
    def ok(data: T | None) -> "ApiResponse[T]":
        return ApiResponse(success=True, message="OK", data=data)

    @staticmethod
    def error(message: str) -> "ApiResponse[None]":
        return ApiResponse(success=False, message=message, data=None)


class PageResult(BaseModel, Generic[T]):
    page: int
    size: int
    total: int
    records: list[T]
