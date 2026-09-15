from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Financial text to analyze for sentiment."
    )

class PredictionResponse(BaseModel):
    sentiment: str =Field(
        ...,
        description="Predicted financial sentiment."
    )

    confidence: float=Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score."
    )    

    explanation: str=Field(
        ...,
        description="LLM-generated explanation of prediction."
    )


