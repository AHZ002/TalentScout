from fastapi import FastAPI

from talentscout.api.routes.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="TalentScout API",
        version="0.1.0",
        description="Agentic AI platform for adaptive technical screening.",
    )

    app.include_router(health_router)

    return app


app = create_app()