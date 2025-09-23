# Sound Realty Inference Service

This scaffold gives you a FastAPI service that:
- serves predictions from your existing model artifacts (`model.pkl`, `model_features.json`)
- joins zipcode-level demographics **on the backend**
- exposes two endpoints:
  - `POST /predict` expects the columns from `future_unseen_examples.csv` (no price/date/id). It performs the zipcode join and returns predictions.
  - `POST /predict_core` accepts only the model's required features from `model_features.json` (bonus endpoint).

## Quickstart (local)

```bash
# 0) ensure you have your artifacts and data copied
ls model/   # should contain model.pkl and model_features.json
ls data/    # should contain zipcode_demographics.csv and future_unseen_examples.csv

# 1) create environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) run the API
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 3) exercise the API
python scripts/test_client.py --url http://localhost:8080 -k 3