import mlflow
from mlflow import MlflowClient
from src.config import config

def promote_candidate():
    """
    Promote the current candidate model version to champion model version
    """

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name=config["mlflow"]["registered_model_name"]

    client=MlflowClient()

    # get the candidate model
    candidate=client.get_model_version_by_alias(name=model_name, alias="candidate")

    candidate_version=candidate.version

    print(f"candidate: {model_name};\n version: {candidate_version}")

    # Assign champion alias to the same model version

    client.set_registered_model_alias(
        name=model_name, alias="champion", version=candidate_version
    )

    print(f"Promoted {model_name}; \n Version {candidate_version} to champion.")

if __name__=="__main__":
    promote_candidate()