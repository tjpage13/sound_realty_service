from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pickle, json, hashlib, os, datetime as dt

APP_NAME = "sound_realty_inference_service"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "model", "model.pkl"))
FEATURES_PATH = os.environ.get("FEATURES_PATH", os.path.join(ROOT_DIR, "model", "model_features.json"))
DEMOGRAPHICS_PATH = os.environ.get("DEMOGRAPHICS_PATH", os.path.join(ROOT_DIR, "data", "zipcode_demographics.csv"))

# Lazy-loaded globals
_model = None
_model_features = None
_demo_df = None
_model_hash = None

def _load_artifacts():
    """Load model, feature list, and demographics (if available)."""
    global _model, _model_features, _demo_df, _model_hash
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            blob = f.read()
            _model_hash = hashlib.sha256(blob).hexdigest()[:12]
            _model = pickle.loads(blob)
    if _model_features is None:
        with open(FEATURES_PATH, "r") as f:
            _model_features = json.load(f)
    if _demo_df is None:
        try:
            demo = pd.read_csv(DEMOGRAPHICS_PATH)
            if "zipcode" in demo.columns:
                demo["zipcode"] = demo["zipcode"].astype(str).str.replace(".0", "", regex=False)
            _demo_df = demo
        except Exception:
            _demo_df = None

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description="List of house rows like future_unseen_examples.csv (no price/date/id)."
    )

class PredictCoreRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description="Records that already match model_features.json exactly."
    )

app = FastAPI(title=APP_NAME)

@app.get("/healthz")
def health():
    try:
        _load_artifacts()
        return {"status": "ok", "model_hash": _model_hash, "feature_count": len(_model_features)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _prepare_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str).str.replace(".0", "", regex=False)
    return df

def _join_demographics(df: pd.DataFrame) -> pd.DataFrame:
    _load_artifacts()
    if _demo_df is None or "zipcode" not in df.columns:
        return df
    return df.merge(_demo_df, on="zipcode", how="left")

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    _load_artifacts()
    # add missing model features as NaN; extra columns are fine
    for col in _model_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[_model_features]

def _predict_df(df: pd.DataFrame) -> np.ndarray:
    _load_artifacts()
    return _model.predict(df.values)

@app.post("/predict")
def predict(req: PredictRequest):
    _load_artifacts()
    try:
        df = _prepare_dataframe(req.records)
        df = _join_demographics(df)
        df = _ensure_features(df)
        preds = _predict_df(df)
        now = dt.datetime.utcnow().isoformat() + "Z"
        return {
            "metadata": {"timestamp_utc": now, "model_hash": _model_hash, "n_predictions": len(preds)},
            "predictions": [float(x) for x in preds],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_core")
def predict_core(req: PredictCoreRequest):
    _load_artifacts()
    try:
        df = pd.DataFrame(req.records)
        df = _ensure_features(df)
        preds = _predict_df(df)
        now = dt.datetime.utcnow().isoformat() + "Z"
        return {
            "metadata": {"timestamp_utc": now, "model_hash": _model_hash, "n_predictions": len(preds)},
            "predictions": [float(x) for x in preds],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))