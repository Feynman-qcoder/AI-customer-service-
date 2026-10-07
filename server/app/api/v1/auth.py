from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser, current_user
from app.db.session import get_session
from app.schemas.auth import AuthTokenResponse, AuthUserResponse, LoginRequest, LogoutRequest, RefreshTokenRequest
from app.schemas.common import ApiResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
service = AuthService()


@router.post("/login")
async def login(request: LoginRequest, session: AsyncSession = Depends(get_session)) -> ApiResponse[AuthTokenResponse]:
    return ApiResponse.ok(await service.login(session, request.username, request.password))


@router.post("/refresh")
async def refresh(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[AuthTokenResponse]:
    return ApiResponse.ok(await service.refresh(session, request.refreshToken))


@router.get("/me")
async def me(user: AuthenticatedUser = Depends(current_user)) -> ApiResponse[AuthUserResponse]:
    return ApiResponse.ok(AuthUserResponse(userId=user.user_id, username=user.username, name=user.name, role=user.role))


@router.post("/logout")
async def logout(request: LogoutRequest) -> ApiResponse[None]:
    await service.logout(request.refreshToken)
    return ApiResponse.ok(None)
