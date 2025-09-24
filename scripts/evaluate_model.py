import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_predict

def load_features(path):
    with open(path, "r") as f:
        obj = json.load(f)
    return obj["features"] if isinstance(obj, dict) and "features" in obj else list(obj)

def normalize_zip(s):
    return (
        s.astype(str)
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"\D", "", regex=True)
         .str.zfill(5)
    )

def align_impute(df, feature_list):
    X = df.reindex(columns=feature_list)
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            mode = X[c].mode(dropna=True)
            X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "").astype(str)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/kc_house_data.csv")
    ap.add_argument("--demo", default="data/zipcode_demographics.csv")
    ap.add_argument("--model", default="model/model.pkl")
    ap.add_argument("--features", default="model/model_features.json")
    ap.add_argument("--target", default="price")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (os.path.exists(args.sales) and os.path.exists(args.demo) and os.path.exists(args.model) and os.path.exists(args.features)):
        raise SystemExit("Missing required file(s). Ensure model + data are present.")

    df = pd.read_csv(args.sales)
    demo = pd.read_csv(args.demo)

    # join demographics
    if "zipcode" not in df.columns or "zipcode" not in demo.columns:
        raise SystemExit("zipcode column missing in one of the CSVs.")
    df["zipcode"] = normalize_zip(df["zipcode"])
    demo["zipcode"] = normalize_zip(demo["zipcode"])
    df = df.merge(demo, on="zipcode", how="left", validate="m:1")

    y = df[args.target].values
    features = load_features(args.features)
    X = align_impute(df, features)

    model = joblib.load(args.model)

    # Holdout
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    mdl = clone(model) if hasattr(model, "get_params") else model
    try:
        mdl.fit(Xtr, ytr)
        ypred_holdout = mdl.predict(Xte)
    except Exception:
        warnings.warn("Model not cloneable/refittable; using loaded model directly on full data for holdout-style eval.")
        ypred_holdout = model.predict(Xte)

    mae_hold = float(mean_absolute_error(yte, ypred_holdout))
    rmse_hold = float(mean_squared_error(yte, ypred_holdout, squared=False))
    r2_hold = float(r2_score(yte, ypred_holdout))

    # 5-fold CV (best effort)
    cv_metrics = {}
    try:
        base = clone(model)
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        ypred_cv = cross_val_predict(base, X, y, cv=kf, n_jobs=None)
        cv_metrics = {
            "mae": float(mean_absolute_error(y, ypred_cv)),
            "rmse": float(mean_squared_error(y, ypred_cv, squared=False)),
            "r2": float(r2_score(y, ypred_cv)),
        }
    except Exception:
        cv_metrics = {"note": "Model not safely cloneable; skipping CV."}

    report = {
        "n_rows": int(len(df)),
        "n_features_model": int(len(features)),
        "metrics": {
            "holdout": {"mae": mae_hold, "rmse": rmse_hold, "r2": r2_hold},
            "cv5": cv_metrics,
        },
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()