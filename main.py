from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import router
from dependencies import get_db_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_service = get_db_service()
    await db_service.init_pool()
    print("Database pool initialized")
    yield
    await db_service.close_pool()
    print("Database pool closed")

app = FastAPI(
    title="AI test service",
    description="A simple service to test AI responses",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}