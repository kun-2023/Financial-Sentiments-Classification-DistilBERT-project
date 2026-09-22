import joblib
from functools import lru_cache
from transformers import pipeline
import torch
from api.settings import (model_path, label_encoder_path)


class SentimentService:
    def __init__(self):
        self.pipeline=self._load_model()
        self.label_encoder=self._load_label_encoder()

    def _load_model(self):
        device=0 if torch.cuda.is_available() else -1

        pipeline_model=pipeline(
            "text-classification", model=str(model_path), tokenizer=str(model_path),
            device=device
        )
        return pipeline_model

    def _load_label_encoder(self):
        if not label_encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at: {label_encoder_path}"
            )
        return joblib.load(label_encoder_path)

    def predict(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Texts can't be empty.")
        
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