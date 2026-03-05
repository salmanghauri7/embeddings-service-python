from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.db.database import db
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to database
    await db.connect()
    yield
    # Shutdown: Disconnect from database
    await db.disconnect()


app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Microservice for PDF embeddings generation",
    version=settings.VERSION,
    lifespan=lifespan
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Embeddings Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected" if db.db else "disconnected"}

