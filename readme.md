#### To Open Mlflow Server
in bash type:
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```  

#### To Run the complete training pipeline from the project root while have mlflow ui open

```bash
python -m src.train_pipeline
```

#### To run inference

python -m src.inference 'Adobe shares fell after earnings' --alias candidate

#### To promote the model to champion from candidate then run the champion version

python -m src.promote
python -m src.inference 'Adobe shares fell after earnings' --alias champion
