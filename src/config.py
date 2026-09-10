from pathlib import Path

project_root=Path(__file__).resolve().parents[1]

config={
    "path":{
        #rea data
        "raw_train":project_root/"data/sent_train.csv",
        "raw_valid":project_root/"data/sent_valid.csv",

        # cleaned data
        "cleaned_data_train":project_root/"outputs/cleaned_data/cleaned_train.csv",
        "cleaned_data_valid":project_root/"outputs/cleaned_data/cleaned_valid.csv",

        # splited data
        "cleaned_split_train": project_root/"outputs/cleaned_split_data/cleaned_split_train.csv",
        "cleaned_split_valid": project_root/"outputs/cleaned_split_data/cleaned_split_valid.csv",
        "cleaned_split_test": project_root/"outputs/cleaned_split_data/cleaned_split_test.csv",

        # embedded data
        "train_embeddings":project_root/"outputs/embedded_data/train.npz",
        "valid_embeddings":project_root/"outputs/embedded_data/valid.npz",
        "test_embeddings":project_root/"outputs/embedded_data/test.npz",

        # encoder_path
        "label_encoder": project_root/ "outputs/encoders/label_encoder.pkl",

        # distilbert_model
        "distilbert_model": project_root/"outputs/nlp_classification_model",

        # distilbert model artifact
        "distilbert_model_artifacts": project_root/"outputs/nlp_classification_model/model",

        # test metrics
        "test_metrics": project_root/"outputs/nlp_classification_model/test_metrics.json"
        },

    "random_state": {
        "random_state": 42,
    },

    "distilbert": {

        "model_id": "distilbert-base-uncased",
        "num_train_epochs": 10,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "early_stopping_patience":2,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "metric_for_best_model": "f1",
    },

    "mlflow": {
        "experiment_name": "financial_sentiments",
        "registered_model_name": "financial_sentiment_distilbert",
        "model_alias": "champion",
        "run_name": "distilbert",
        "tracking_uri": "http://127.0.0.1:5000",
        "artifact_uri": (project_root/"mlartifacts").as_uri()
    },

    "llm": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "max_new_tokens": 30,
        "do_sample": False,
    },
}