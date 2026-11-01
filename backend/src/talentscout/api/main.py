from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from talentscout.api.routes.health import router as health_router
from talentscout.config.settings import get_settings
from talentscout.jobs.api.documents import router as documents_router
from talentscout.jobs.api.interviews import router as interviews_router
from talentscout.jobs.api.routes import router as jobs_router


def create_app() -> FastAPI:
    # Load the validated application configuration.
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic AI platform for adaptive technical screening.",
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(documents_router)
    app.include_router(interviews_router)

    return app


app = create_app()
