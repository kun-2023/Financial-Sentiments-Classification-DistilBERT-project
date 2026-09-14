from pathlib import Path
import os
from dotenv import load_dotenv

base_dir=Path(__file__).resolve().parent.parent
load_dotenv(base_dir/".env")

mlflow_tracking_uri=os.environ["mlflow_tracking_uri"]
registered_model_name=os.environ["registered_model_name"]
champion_alias=os.environ["champion_alias"]

label_encoder_path=(base_dir/os.environ["label_encoder_path"])
llm_model_name=os.environ["llm_model_name"]
log_level=os.getenv("log_level", "INFO").upper()