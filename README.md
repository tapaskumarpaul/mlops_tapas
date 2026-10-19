# CardioRisk MLOps

End-to-end educational MLOps implementation for binary heart-disease risk classification using the UCI Cleveland Heart Disease dataset.

> **Safety:** This project is for assessment only. It is not a medical device and must not be used for diagnosis or treatment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_eda.py
python scripts/train.py
uvicorn src.cardiorisk.api:app --host 0.0.0.0 --port 8000
```

Open API documentation at `http://localhost:8000/docs`, health at `http://localhost:8000/health`, and metrics at `http://localhost:8000/metrics`.

## Docker
```bash
docker build -t cardiorisk-api:1.0.0 .
docker run --rm -p 8000:8000 cardiorisk-api:1.0.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @sample_request.json
```

## Minikube
```bash
minikube start
minikube image load cardiorisk-api:1.0.0
kubectl apply -f deployment/namespace.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl port-forward -n cardiorisk service/cardiorisk-api 8000:80
```
Then use `http://localhost:8000`. This repository intentionally provides local access instructions rather than claiming a public deployment.

## Tests and tracking
```bash
pytest -q --cov=src
mlflow ui --backend-store-uri ./mlruns
```
MLflow UI: `http://127.0.0.1:5000`.

## Data attribution
UCI Machine Learning Repository, Heart Disease dataset, DOI: 10.24432/C52P4X, CC BY 4.0. The download script retrieves the processed Cleveland file and creates `data/processed/heart_disease_clean.csv`.
