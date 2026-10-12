"""FastAPI application entrypoint and middleware configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.api import api_router
from backend.app.core.config import backend_settings
from backend.app.core.logging import setup_logging
from backend.app.db.base import Base
from backend.app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Execute startup and shutdown lifecycle hooks."""
    setup_logging()
    # Ensure database tables exist (useful for SQLite / initial setups)
    Base.metadata.create_all(bind=engine)
    yield


def create_application() -> FastAPI:
    """Application factory for FastAPI server."""
    app = FastAPI(
        title=backend_settings.PROJECT_NAME,
        version="0.1.0",
        openapi_url=f"{backend_settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=backend_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(api_router, prefix=backend_settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    def root_redirect() -> dict[str, str]:
        return {
            "name": backend_settings.PROJECT_NAME,
            "version": "0.1.0",
            "docs": "/docs",
            "health": f"{backend_settings.API_V1_STR}/health",
        }

    return app


app = create_application()
