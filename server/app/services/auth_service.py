from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import UnauthorizedError
from app.core.security import create_access_token, create_access_token_claims, hash_password, verify_password
from app.db.models import UserAccount
from app.schemas.auth import AuthTokenResponse, AuthUserResponse
from app.services.refresh_token_service import RefreshTokenRecord, refresh_token_service


class AuthService:
    async def login(self, session: AsyncSession, username: str, password: str) -> AuthTokenResponse:
        result = await session.execute(select(UserAccount).where(UserAccount.username == username))
        user = result.scalar_one_or_none()
        if user is None or user.status != "ACTIVE" or not verify_password(password, user.password_hash):
            # Demo fallback for a freshly migrated database before seed data is inserted.
            if username == settings.demo_customer_username and password == settings.demo_customer_password:
                user = await self._ensure_demo_user(session, username, password, "演示用户", "CUSTOMER")
            elif username == settings.demo_admin_username and password == settings.demo_admin_password:
                user = await self._ensure_demo_user(session, username, password, "后台管理员", "ADMIN")
            else:
                raise UnauthorizedError("账号或密码错误")
        user.last_login_at = datetime.now()
        await session.commit()
        refresh_token = await refresh_token_service.issue(
            RefreshTokenRecord(
                user_id=user.id,
                username=user.username,
                display_name=user.display_name,
                role=user.role,
            )
        )
        return AuthTokenResponse(
            accessToken=create_access_token(user),
            refreshToken=refresh_token,
            expiresIn=settings.access_token_ttl_minutes * 60,
            user=AuthUserResponse(userId=user.id, username=user.username, name=user.display_name, role=user.role),
        )

    async def refresh(self, session: AsyncSession, refresh_token: str) -> AuthTokenResponse:
        record = await refresh_token_service.consume(refresh_token)
        if record is None:
            raise UnauthorizedError("刷新令牌无效或已过期")
        user = await session.get(UserAccount, record.user_id)
        if user is None or user.status != "ACTIVE":
            raise UnauthorizedError("账号不可用")
        next_refresh_token = await refresh_token_service.issue(
            RefreshTokenRecord(
                user_id=user.id,
                username=user.username,
                display_name=user.display_name,
                role=user.role,
            )
        )
        return AuthTokenResponse(
            accessToken=create_access_token_claims(user.id, user.username, user.display_name, user.role),
            refreshToken=next_refresh_token,
            expiresIn=settings.access_token_ttl_minutes * 60,
            user=AuthUserResponse(userId=user.id, username=user.username, name=user.display_name, role=user.role),
        )

    async def logout(self, refresh_token: str | None) -> None:
        if refresh_token:
            await refresh_token_service.revoke(refresh_token)

    async def _ensure_demo_user(
        self,
        session: AsyncSession,
        username: str,
        password: str,
        display_name: str,
        role: str,
    ) -> UserAccount:
        result = await session.execute(select(UserAccount).where(UserAccount.username == username))
        user = result.scalar_one_or_none()
        if user is None:
            user = UserAccount(
                username=username,
                        password_hash=hash_password(password),
                display_name=display_name,
                role=role,
                status="ACTIVE",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(user)
            await session.flush()
        return user
