#### To Open Mlflow Server
in bash type:
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --port 5000
```  