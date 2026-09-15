from functools import lru_cache
from dotenv import load_dotenv
import joblib
import mlflow
import mlflow.transformers
import torch
from api.settings import (registered_model_name, 
                          champion_alias,
                          mlflow_tracking_uri,
                          label_encoder_path)

mlflow.set_tracking_uri(mlflow_tracking_uri)
model_uri=(f"models:/{registered_model_name}@{champion_alias}")

class SentimentService:
    def __init__(self):
        self.pipeline=self._load_model()
        self.label_encoder=self._load_label_encoder()

    def _load_model(self):
        device=0 if torch.cuda.is_available() else -1

        pipeline=mlflow.transformers.load_model(
            model_uri,
            return_type="pipeline",
            device=device
        )
        return pipeline

    def _load_label_encoder(self):
        if not label_encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at: {label_encoder_path}"
            )
        return joblib.load(label_encoder_path)

    def predict(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Texts cann't be empty.")
        
        result=self.pipeline(text.strip())
        prediction=result[0]
        sentiment=str(prediction["label"])
        confidence=float(prediction["score"])

        return str(sentiment), confidence

@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    return SentimentService()

def predict_sentiment(text: str) -> tuple[str, float]:
    service=get_sentiment_service()
    return service.predict(text)