from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.services.sentiment_service import get_sentiment_service
from api.services.explanation_service import get_explanation_service

from api.routers.predict import router as predict_router
from api.routers.health import router as health_router

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Loading sentiment model..")
    get_sentiment_service()

    print("Load explanation model...")
    get_explanation_service()

    print("Models ready.")

    yield


app=FastAPI(
    title="Financial Sentiment API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(predict_router)