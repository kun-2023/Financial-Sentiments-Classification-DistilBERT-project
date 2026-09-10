from src.preprocess import run_preprocessing
from src.train import train_model


def run_pipeline():

    print("Starting preprocessing...")
    run_preprocessing()

    print("Starting model training...")
    train_model()

    print("Training pipeline complete.")


if __name__ == "__main__":
    run_pipeline()