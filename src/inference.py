import argparse
import mlflow
import mlflow.transformers
import torch
from src.config import config
from src.preprocess import clean_text
import numpy as np


def load_model(alias="champion"):
    """
    Load a registered DistilBert model from MLflow Model Reegistry
    """

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_registry_uri(config["mlflow"]["tracking_uri"])
    model_name=config["mlflow"]["registered_model_name"]

    model_uri=f"models:/{model_name}@{alias}"
    device=0 if torch.cuda.is_available() else None

    model=mlflow.transformers.load_model(
        model_uri=model_uri,
        return_type="pipeline",
        device=device
    )

    print(f"Loaded {model_name}@{alias}")

    return model

def predict_sentiment(text, model):
    """preprocessing and return prediction with confidence."""
    cleaned_text=clean_text(text)

    if not cleaned_text:
        raise ValueError("Text is empty after preprocessing.")
    result=model(cleaned_text)[0]

    prediction={
        "text": text,
        "cleaned_text":cleaned_text,
        "sentiment": result["label"],
        "confidence": float(result["score"])
    }

    return prediction


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "text",
        type=str,
        help="Financial text to classify"
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="MLflow model alias: champion or candidate"

    )

    args=parser.parse_args()
    model=load_model(alias=args.alias)

    prediction=predict_sentiment(
        text=args.text, model=model
    )

    print("\nPrediction:")
    print(f"Text: {prediction["text"]}")
    print(f"Cleaned: {prediction["cleaned_text"]}")
    print(f"Sentiment: {prediction["sentiment"]}")
    print(f"Confidence: {prediction["confidence"]*100:.2f}")

if __name__=="__main__":
    main()