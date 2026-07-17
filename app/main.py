from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.api.v1.routes import router
from app.core.config import get_settings
from app.services.inference import engine
from app.services.orthanc_poller import poller_loop
import asyncio

log = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("sonoai_starting", env=settings.APP_ENV)
    engine.load_models()
    log.info("sonoai_ready")
    poller_task = asyncio.create_task(poller_loop())
    yield
    poller_task.cancel()
    log.info("sonoai_shutting_down")


app = FastAPI(
    title="SonoAI API",
    description="AI-assisted ultrasound interpretation for East Africa",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.APP_DEBUG else None,
    redoc_url="/redoc" if settings.APP_DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.APP_DEBUG else ["https://app.sonoai.africa"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"service": "SonoAI", "status": "running", "version": "1.0.0"}
