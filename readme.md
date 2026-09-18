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
  --host 0.0.0.0 \
  --port 5000
  --workers 1
```
##### then manually open http://localhost:5000

```bash
# fastapi http://127.0.01:8000
uvicorn api.main:app --reload
# then add /docs at the end
# streamlit http://localhost:8501
streamlit run frontend/app.py
```

#### To Build Docker Image
```bash
docker compose up --build
```

#### Running the app locally
##### The application uses the following:
* MLflow for model registry and artifact storage
* FastAPI for the prediction API
* Streamlit for the frontend
* Docker Compose to run the FastAPI and Streamlit services

#### Steps
1. Activate the virtual environment
```bash
source .senti_venv/Scripts/activate
```
2. Start MLflow server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
2.1 then manually type http://localhost:5000

3. start the application with Docker Compose in a second terminal
```bash
docker compose up -d
```
3.1 if the image needs to be rebuilt
```bash
docker compose up --build -d
```
4. Check the container
```bash
docker compose ps
```
5. wait for the model to load then ctrl+c to get out
```bash
docker compose logs -f api
```
5.1 wait to see(this may take a while):
```
Models ready
Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

6. check API health and expect {"status":"ok"} before running the app
```bash
curl http://localhost:8000/health
```
7. Open the Streamlit application
```bash
start http://localhost:8501
```
8. shut down application
8.1 shut down docker
```bash
docker compose down
```
8.2 stop MLflow
```bash
Ctrl+C
```
#### torun pytest
```bash
pytest -v
```