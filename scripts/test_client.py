import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/future_unseen_examples.csv")
    ap.add_argument("--url", default=os.getenv("PREDICT_URL", "http://localhost:8080/predict"))
    ap.add_argument("--limit", type=int, default=5, help="Send first N rows for a quick smoke test.")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    payload = {"records": df.to_dict(orient="records")}
    t0 = time.time()
    r = requests.post(args.url, json=payload, timeout=30)
    dt = (time.time() - t0) * 1000
    r.raise_for_status()
    out = r.json()

    print(json.dumps({
        "request_size": len(payload["records"]),
        "latency_ms": int(dt),
        "response": out
    }, indent=2))

if __name__ == "__main__":
    main()