"""
visp.main
~~~~~~~~~
FastAPI application factory and entry point.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.api.routes import stream, events, health
from backend.core.config import get_settings
from backend.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Real-time video behavior analysis for workplace and retail safety. "
            "Violence detection · PPE compliance · Zone intrusion alerting."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Prometheus metrics ─────────────────────────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Routes ────────────────────────────────────────────────────
    app.include_router(stream.router)
    app.include_router(events.router)
    app.include_router(health.router)

    # ── Startup / shutdown ────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("VISP backend starting up (model=%s)", settings.model_backend.value)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("VISP backend shutting down")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
