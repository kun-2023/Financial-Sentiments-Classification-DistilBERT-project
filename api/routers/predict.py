from fastapi import APIRouter, HTTPException
from api.schemas.prediction import (
    PredictionRequest, PredictionResponse)
from api.services.sentiment_service import predict_sentiment
from api.services.explanation_service import generate_explanation