from fastapi import APIRouter, HTTPException
from api.schemas.prediction import (
    PredictionRequest, PredictionResponse)
from api.services.sentiment_service import predict_sentiment
from api.services.explanation_service import generate_explanation

router=APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

@router.post(
    "",
    response_model=PredictionResponse,
)

def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        sentiment, confidence=predict_sentiment(request.text)

        explanation=generate_explanation(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
        )

        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence,
            explanation=explanation,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="prediction failed.",
        ) from e
