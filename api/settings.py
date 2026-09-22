from pathlib import Path
import os
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

model_path = base_dir / os.getenv(
    "model_path",
    "outputs/nlp_classification_model/model"
)

label_encoder_path = base_dir / os.getenv(
    "label_encoder_path",
    "outputs/encoders/label_encoder.pkl"
)

llm_model_name = os.getenv(
    "llm_model_name",
    "HuggingFaceTB/SmolLM2-360M-Instruct"
)

log_level = os.getenv("log_level", "INFO").upper()