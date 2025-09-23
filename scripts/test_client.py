import pandas as pd
import requests
import argparse
import json

def main(url: str, k: int):
    df = pd.read_csv("data/future_unseen_examples.csv").head(k)
    records = df.to_dict(orient="records")
    resp = requests.post(f"{url}/predict", json={"records": records}, timeout=30)
    print("Status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    main(args.url, args.k)