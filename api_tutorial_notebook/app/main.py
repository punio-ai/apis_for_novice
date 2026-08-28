from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
from app.database import init_db
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("✅ Database initialized")
    yield
    print("👋 Shutting down")

app = FastAPI(
    title="Knowledge API",
    description="Production-grade FastAPI foundation for RAG.",
    version="0.2.0",
    lifespan=lifespan
)

# 1. Global Exception Handler for Pydantic Validation Errors


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Failed",
            "details": exc.errors(),
            "message": "Please check your request payload."
        },
    )

# 2. Global Exception Handler for General Errors


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error",
                 "message": "An unexpected error occurred."}
    )

# 3. Custom Middleware for Request Timing


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log slow requests (e.g., > 1 second)
    if process_time > 1.0:
        print(
            f"⚠️ SLOW REQUEST: {request.method} {request.url.path} took {process_time:.2f}s")

    return response

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Knowledge API v0.2.0 is running", "docs": "/docs"}
