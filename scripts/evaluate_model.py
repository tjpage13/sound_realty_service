import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json, pickle, os, argparse

def load_artifacts(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "model_features.json"), "r") as f:
        features = json.load(f)
    return model, features

def evaluate(data_dir, model_dir):
    model, features = load_artifacts(model_dir)
    sales = pd.read_csv(os.path.join(data_dir, "kc_house_data.csv"))
    demos = pd.read_csv(os.path.join(data_dir, "zipcode_demographics.csv"))

    # Normalize zipcode to string for join
    for df in (sales, demos):
        if "zipcode" in df.columns:
            df["zipcode"] = df["zipcode"].astype(str).str.replace(".0", "", regex=False)

    df = sales.merge(demos, on="zipcode", how="left")

    y = df["price"].values
    X = df.reindex(columns=features).values

    # Simple holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    results = {
        "train": metrics(y_train, preds_train),
        "test": metrics(y_test, preds_test),
    }

    # Cross-val on full dataset (if estimator supports)
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
        r2_scores = cross_val_score(model, X, y, scoring="r2", cv=kf)
        results["cv"] = {
            "RMSE_mean": float(rmse_scores.mean()),
            "RMSE_std": float(rmse_scores.std()),
            "R2_mean": float(r2_scores.mean()),
            "R2_std": float(r2_scores.std()),
        }
    except Exception as e:
        results["cv"] = {"error": str(e)}

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_dir", default="model")
    args = parser.parse_args()
    res = evaluate(args.data_dir, args.model_dir)
    print(json.dumps(res, indent=2))