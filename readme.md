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
```bash
python -m src.inference 'Adobe shares fell after earnings' --alias candidate
```
#### To promote the model to champion from candidate then run the champion version

python -m src.promote
python -m src.inference 'Adobe shares fell after earnings' --alias champion

#### Reproduce model with DVC
```bash
dvc repro
# after model is being trained and registed as candidate. promote it to champion
python -m promote.py
```

#### To show FastAPI Swagger
```bash
# turn on mlflow ui
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```
# turn on fastapi swagger
```bash
uvicorn api.main:app --reload
```
# then go to docs by adding /docs to http://127.0.0.1:8000
```
http://127.0.0.1:8000/docs
```

#### Run frontend by turning on mlflow ui, fastapi swagger, and streamlit 
```bash
# mlflow http://127.0.01:5000
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000

# fastapi http://127.0.01:8000
uvicorn api.main:app --reload

# streamlit http://localhost:8501
streamlit run frontend/app.py
```