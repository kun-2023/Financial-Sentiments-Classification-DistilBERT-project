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

    confidence: str=Field(
        ...,
        description="Model confidence score as a percentage."
    )    

    explanation: str=Field(
        ...,
        description="LLM-generated explanation of prediction."
    )


