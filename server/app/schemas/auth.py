from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    displayName: str


class AuthUserResponse(BaseModel):
    userId: int
    username: str
    name: str
    role: str


class AuthTokenResponse(BaseModel):
    accessToken: str
    refreshToken: str | None = None
    expiresIn: int
    user: AuthUserResponse


class RefreshTokenRequest(BaseModel):
    refreshToken: str


class LogoutRequest(BaseModel):
    refreshToken: str | None = None
