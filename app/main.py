import json
import os
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_NAME = "sound-realty-service"

# ---------- Config (override via env if you need to) ----------
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of /app)

MODEL_PATH = os.getenv("MODEL_PATH", str(APP_ROOT / "model" / "model.pkl"))
FEATURES_PATH = os.getenv("FEATURES_PATH", str(APP_ROOT / "model" / "model_features.json"))
DEMOGRAPHICS_PATH = os.getenv("DEMOGRAPHICS_PATH", str(APP_ROOT / "data" / "zipcode_demographics.csv"))

# ---------- Pydantic payloads ----------
class PredictRequest(BaseModel):
    # rows look like data/future_unseen_examples.csv (NO demo cols)
    records: List[Dict[str, Any]] = Field(default_factory=list)

class PredictCoreRequest(BaseModel):
    # rows contain exactly the model's training features
    records: List[Dict[str, Any]] = Field(default_factory=list)

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str
    model_features: List[str]
    n_records: int
    timing_ms: int

# ---------- App ----------
app = FastAPI(title=APP_NAME)
_model = None
_model_features: List[str] = []
_demo_df: Optional[pd.DataFrame] = None
_model_version: str = "unknown"

def _load_artifacts():
    global _model, _model_features, _demo_df, _model_version
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    _model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feats = json.load(f)
    # Accept either {"features": [...]} or a plain list
    _model_features = feats["features"] if isinstance(feats, dict) and "features" in feats else list(feats)

    if not os.path.exists(DEMOGRAPHICS_PATH):
        raise FileNotFoundError(f"Missing demographics file: {DEMOGRAPHICS_PATH}")
    _demo_df = pd.read_csv(DEMOGRAPHICS_PATH)

    # crude version hash from file mtimes/sizes
    m_stat = os.stat(MODEL_PATH)
    f_stat = os.stat(FEATURES_PATH)
    _model_version = f"{int(m_stat.st_mtime)}-{m_stat.st_size}-{f_stat.st_size}"

def _as_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)

def _normalize_zip(df: pd.DataFrame, col: str = "zipcode") -> pd.DataFrame:
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
    # to string and zero-fill 5 (handles ints, floats, strings)
    z = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True).str.zfill(5)
    df = df.copy()
    df[col] = z
    return df

def _join_demographics(df: pd.DataFrame) -> pd.DataFrame:
    assert _demo_df is not None
    left = _normalize_zip(df, "zipcode")
    right = _demo_df.copy()
    # ensure zipcode is str zfilled in demographics too
    if "zipcode" in right.columns:
        right["zipcode"] = right["zipcode"].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True).str.zfill(5)
    joined = left.merge(right, on="zipcode", how="left", validate="m:1")
    return joined

def _align_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    # keep only required features; add any missing with NaN
    X = df.reindex(columns=_model_features, fill_value=np.nan)
    # simple, safe imputations (columnwise): numeric -> median, other -> mode
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            med = X[c].median()
            X[c] = X[c].fillna(med)
        else:
            mode = X[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else ""
            X[c] = X[c].fillna(fill).astype(str)
    return X

@app.on_event("startup")
def _startup():
    _load_artifacts()

@app.get("/healthz")
def healthz():
    ok = _model is not None and len(_model_features) > 0
    return {
        "status": "ok" if ok else "not_ready",
        "model_version": _model_version,
        "n_features": len(_model_features),
        "features_head": _model_features[:5],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Accepts rows like data/future_unseen_examples.csv (NO demographics).
    We add zipcode demographics on the backend, align to model features, and predict.
    """
    t0 = time.time()
    df = _as_df(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")
    df = _join_demographics(df)
    X = _align_and_impute(df)
    try:
        yhat = _model.predict(X)  # works for sklearn regressors; extend if proba, etc.
        preds = [float(v) for v in np.asarray(yhat).ravel().tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    ms = int((time.time() - t0) * 1000)
    return PredictResponse(
        predictions=preds,
        model_version=_model_version,
        model_features=_model_features,
        n_records=len(df),
        timing_ms=ms,
    )

@app.post("/predict_core", response_model=PredictResponse)
def predict_core(payload: PredictCoreRequest):
    """
    Bonus endpoint: callers provide exactly the training features (no joining).
    """
    t0 = time.time()
    df = _as_df(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided.")
    X = _align_and_impute(df)
    try:
        yhat = _model.predict(X)
        preds = [float(v) for v in np.asarray(yhat).ravel().tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    ms = int((time.time() - t0) * 1000)
    return PredictResponse(
        predictions=preds,
        model_version=_model_version,
        model_features=_model_features,
        n_records=len(df),
        timing_ms=ms,
    )