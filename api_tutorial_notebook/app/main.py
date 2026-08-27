from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import init_db
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables
    await init_db()
    print("✅ Database initialized")
    yield
    # Shutdown: Cleanup (if needed)
    print("👋 Shutting down")

app = FastAPI(
    title="Knowledge API",
    description="A personal knowledge base API. Foundation for RAG integration.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Knowledge API is running",
        "docs": "/docs",
        "status": "healthy"
    }
