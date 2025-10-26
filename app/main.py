from fastapi import FastAPI

from app.api.v1.endpoints import router as v1_router
from app.core.config import settings

app = FastAPI(title=settings.project_name, version=settings.api_version, debug=settings.debug)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": settings.api_version}


app.include_router(v1_router, prefix="/api/v1")
