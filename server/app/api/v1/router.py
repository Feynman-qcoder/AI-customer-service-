from fastapi import APIRouter

from app.api.v1 import admin_agent, admin_misc, auth, chat, commerce, conversations, documents, health, tickets

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(admin_agent.router)
api_router.include_router(admin_misc.router)
api_router.include_router(commerce.router)
api_router.include_router(conversations.router)
api_router.include_router(documents.router)
api_router.include_router(documents.public_router)
api_router.include_router(chat.router)
api_router.include_router(tickets.router)
api_router.include_router(health.router)
