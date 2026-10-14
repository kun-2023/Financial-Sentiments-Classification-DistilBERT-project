## End-to-End Financial Texts Sentiment Classification
### Project Overview
* App in a glance: This app is to classify the sentiment of a financial twitts or comments from analysts as positive, negative, or neutral. The app deploys a **fine-tuned DistilBert** model for sentiment classification and a **small language model** to explain the reason behind the sentiment.
* Model: 1. A finetuned pre-trained transformer **DistilBert**. It offers high accuracies and reduces training times. 2. A small language model: SmoLM2 for explaining the predictions.
* Techstacks: **Python**, **Pytorch**, **DistilBert**, **MLflow**, **DVC**, **AWS ECR**, **AWS ECS**, **Docker**, **FastAPI**, **Streamlit**, **CI/CD pipeline**, **DVC Retraining pipeline**
* Highlights: The app was integrated with 2 pipelines. One is **DVC retraining pipeline**, which would retrain the model generating model artifacts, metrics dataframes by simply commiting `dvc repro` in bash; the other is **CI/CD pipeline**, which allows the continous integration(install depedencies, run tests) of new codes and continous deployment(log in AWS, build, tag and push docker images, deploy the app) of the app with a line of code `git push`. It saves time and reduce mistakes.
* Be aware: The model needs compatiable GPU for training. It also relies on MLflow.

### Testing Metrics
|Metrics|Scores|
|-------|------|
|Accuracy|86.34%|
|Precision|81.83%|
|Recall|82.29%|
|F1|82.21%|

### Run App locally with docker
```bash

# 1 clone the repo to that folder
git clone https://github.com/kun-2023/Financial-Sentiments-Classification-DistilBERT-project financial_sentiments

# 2 navigate to project directory
cd financial_sentiments


# 3 create and activate virtual environment
python -m venv .senti_venv
source .senti_venv/Scripts/activate
# 4 install dependencies
pip install -r requirements.txt
# 4.1 clone .env file
cp .env.example .env
# 5 download data and model from AWS s3
dvc pull
# 6 Start MLflow in a seperate terminal
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1
# 6 build docker images
docker compose up --build -d
# 7 type http://localhost:8501 in chrome browser or any browser of your choice
# Afterwards, remove docker containers and networks
docker compose down
```


### How to retrain a model automatically

```bash
# 1 Turn on mlflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1

# 2 Retrain with DVC
dvc repro

# 3 then promote it to champion manually
python -m src.promote
```  

`Thanks for Reading`