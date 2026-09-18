import mlflow
from mlflow import MlflowClient

track_uri="http://127.0.0.1:5000"
model_name="financial_sentiment_distilbert"

mlflow.set_tracking_uri(track_uri)

mlflow.set_experiment("financial_sentiment_docker")

with mlflow.start_run():
    model_info=mlflow.transformers.log_model(
        transformers_model="outputs/nlp_classification_model/model",
        name="model",
        task="text-classification",
        registered_model_name=model_name
    )
    print("model uri:", model_info.model_uri)

client=MlflowClient()

versions=client.search_model_versions(
    f"name='{model_name}'"
)

latest_version=max(
    versions, key=lambda x: int(x.version)
)

client.set_registered_model_alias(
    name=model_name, alias="champion", version=latest_version.version
)
print(f"New champion {latest_version.version}")