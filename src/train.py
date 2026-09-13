import json
import random
import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import joblib
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    )
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from mlflow import MlflowClient

from src.config import config
seed=config["random_state"]["random_state"]

# setting up seed
def set_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load prepared data
def load_data():
    train_df=pd.read_csv(config["path"]["cleaned_split_train"])        
    valid_df=pd.read_csv(config["path"]["cleaned_split_valid"])        
    test_df=pd.read_csv(config["path"]["cleaned_split_test"])        

    train_dataset=Dataset.from_dict({
    "text": train_df["cleaned_text"].tolist(),
    "label": train_df["senti_label"].tolist()
    })

    test_dataset=Dataset.from_dict({
    "text": test_df["cleaned_text"].tolist(),
    "label": test_df["senti_label"].tolist()
    })

    valid_dataset=Dataset.from_dict({
    "text": valid_df["cleaned_text"].tolist(),
    "label": valid_df["senti_label"].tolist()
    })

    dataset=DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset,
    "test": test_dataset,
    })

    return dataset


# Metrics

def compute_metrics(eval_pred):
    logits, labels=eval_pred
    preds=np.argmax(logits, axis=1)
    metrics={
        "accuracy": accuracy_score(
            labels, 
            preds
            ),

        "precision": precision_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        )
    }


    return metrics


# Tokenization

def tokenize_data(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_dataset=dataset.map(
        tokenize, batched=True
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ],

        output_all_columns=True
    )

    return tokenized_dataset



# Build model and tokenizer

def build_model(dataset):
    # set label and ids
    label_encoder=joblib.load(config["path"]["label_encoder"])

    id2label={
        i: str(label) for i, label in enumerate(label_encoder.classes_)
    }

    label2id={label: i for i, label in id2label.items()}
    model_id=config["distilbert"]["model_id"]

    tokenizer=AutoTokenizer.from_pretrained(model_id)
    num_labels=len(label_encoder.classes_)
    model=AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return model, tokenizer

# Build trainer
def build_trainer(
        model,
        tokenizer,
        tokenized_dataset
):
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

    training_args=TrainingArguments(
        output_dir=config["path"]["distilbert_model"],
        num_train_epochs= config["distilbert"]["num_train_epochs"],
        per_device_train_batch_size=config["distilbert"]["train_batch_size"],
        per_device_eval_batch_size=config["distilbert"]["eval_batch_size"],
        learning_rate=config["distilbert"]["learning_rate"],
        weight_decay=config["distilbert"]["weight_decay"],

        eval_strategy=config["distilbert"]["eval_strategy"],
        save_strategy=config["distilbert"]["save_strategy"],
        logging_strategy=config["distilbert"]["logging_strategy"] ,

        load_best_model_at_end=True,
        metric_for_best_model=config["distilbert"]["metric_for_best_model"],
        greater_is_better=True,
        report_to="none",
        seed=seed)

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["distilbert"]["early_stopping_patience"]
            )
        ]

    )
    return trainer

# Train Model

def train_model():
    # set_random_seed
    set_seed()
    #load datasets
    dataset=load_data()
    # build model and tokenizer
    model, tokenizer=build_model(dataset)
    # tokenize dataset
    tokenized_dataset=tokenize_data(dataset, tokenizer)
    # build trainer
    trainer=build_trainer(model, tokenizer, tokenized_dataset)

    # mlflow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_registry_uri(config["mlflow"]["tracking_uri"])
    experiment_name=config["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)    
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        # log parameters
        mlflow.log_params({
            "model": config["distilbert"]["model_id"],
            "epochs": config["distilbert"]["num_train_epochs"],
            "batch_size": config["distilbert"]["train_batch_size"],
            "learning_rate": config["distilbert"]["learning_rate"],
            "weight_decay": config["distilbert"]["weight_decay"],
            "early_stopping_patience": config["distilbert"]["early_stopping_patience"]
        })

        trainer.train()

        
        # evaluate test dataset
        test_metrics=trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")

        # log test metrics
        mlflow.log_metrics({
            "test_accuracy": test_metrics["test_accuracy"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
            "test_f1": test_metrics["test_f1"]
        })

        # save model and tokenizer locally
        trainer.save_model(
            config["path"]["distilbert_model_artifacts"]
        )

        tokenizer.save_pretrained(
            config["path"]["distilbert_model_artifacts"]
        )

        # log model to mlflow
        model_info=mlflow.transformers.log_model(
            transformers_model={
                "model": trainer.model,
                "tokenizer": tokenizer
            },
            name="distilbert_model",
            task="text-classification",
            registered_model_name=config["mlflow"]["registered_model_name"]
        )

        

        # promote the model to candidate
        client=MlflowClient()

        client.set_registered_model_alias(
            name=config["mlflow"]["registered_model_name"],
            alias=config["mlflow"]["candidate_alias"],
            version=str(model_info.registered_model_version)
        )

        print("model_uri: ", model_info.model_uri)
        print("Run_id: ", model_info.run_id)
        print("Registered_model_version: ",model_info.registered_model_version)

    #save metrics
    metrics = {
        "accuracy": test_metrics["test_accuracy"],
        "precision": test_metrics["test_precision"],
        "recall": test_metrics["test_recall"],
        "f1": test_metrics["test_f1"]
    }

    config["path"]["test_metrics"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(config["path"]["test_metrics"],"w") as f:
        json.dump(metrics,f,indent=4)
            

    print("Training Complete")
    print("\nTest metrics:")
    print(test_metrics)

# Run training

if __name__=="__main__":
    train_model()
