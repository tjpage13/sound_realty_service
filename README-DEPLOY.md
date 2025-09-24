# Deployment Guide – Sound Realty Service

This guide explains how to build, run, and test the **Sound Realty Service** API.

## 1. Build the Docker image

From the project root:

    docker build -t sound-realty-service .

## 2. Run the service

Run the container, exposing port 8080:

    docker run -p 8080:8080 sound-realty-service

By default, the service looks for:

- Model file: ./model/model.pkl  
- Features file: ./model/model_features.json

You can override these with environment variables:

    docker run -p 8080:8080 \
      -e MODEL_PATH=/models/alt_model.pkl \
      -e FEATURES_PATH=/models/alt_features.json \
      sound-realty-service

## 3. API Endpoints

- GET /healthz  
  Returns a simple status message. Example response:  
      {"status": "ok"}

- POST /predict  
  Main prediction endpoint.  
  - Input: Records with the schema of future_unseen_examples.csv.  
  - Behavior: Service automatically joins zipcode demographics.  
  - Output: Predictions plus metadata, for example:  
        {
          "predictions": [250000.0, 315000.0],
          "model_version": "20230925T123456",
          "model_features": ["beds", "baths", "sqft", "zipcode_income", "..."],
          "n_records": 2,
          "timing_ms": 42
        }

- POST /predict_core  
  Bonus endpoint.  
  - Input: Records with the full model feature set already prepared (no demographics join).  
  - Output: Same format as /predict.

## 4. Test the service

A test client script is provided. It reads data/future_unseen_examples.csv and submits rows to the API.

Example usage:

    python scripts/test_client.py --url http://localhost:8080/predict --limit 3

This sends 3 rows to the /predict endpoint and prints the response.

## 5. Evaluating the Model

The repository includes a script to evaluate model performance on holdout and cross-validation splits:

    python scripts/evaluate_model.py

This reports metrics including MAE, RMSE, and R².

## 6. Deployment Notes

- Scaling  
  Run multiple replicas behind a load balancer or API Gateway.  
  Example: deploy on Kubernetes with an HPA (Horizontal Pod Autoscaler).

- Model versioning  
  Each response includes a model_version field (based on file metadata).  
  To deploy a new model, replace the model file and restart the service.  
  The MODEL_PATH and FEATURES_PATH environment variables allow hot-swapping models without code changes.

- Updated models  
  When iterating on the model, prefer traditional machine learning algorithms (e.g., linear models, tree-based methods) rather than deep learning.  
  The goal is to deliver a robust “80% solution” that balances performance and maintainability.  
  To deploy an updated model, train and save it to a new file (for example, `model_v2.pkl`), then restart the service with:  

      docker run -p 8080:8080 \
        -e MODEL_PATH=/models/model_v2.pkl \
        -e FEATURES_PATH=/models/model_features_v2.json \
        sound-realty-service

  This allows swapping in new models without code changes or redesigning the API.

- Health checks  
  Use /healthz for readiness and liveness probes in orchestration platforms.

## 7. Future Suggestions

Based on current evaluation results:

- **Model performance:**  
  - Test MAE ≈ $102,337 on a dataset with median sale price ≈ $450,000.  
    - → About 22.7% error relative to median price (slightly above the 15–20% “nice to have” band).  
  - Test RMSE ≈ $202,625 vs. market standard deviation ≈ $367,127.  
    - → Error is ~55% of natural market variability, showing the model is capturing useful signal.  
  - Test R² ≈ 0.73, which indicates solid explanatory power.

- **Opportunities for improvement:**  
  - Try alternative traditional ML algorithms (e.g., gradient boosting, random forest, regularized linear models) to balance bias/variance.  
  - Incorporate additional features (e.g., interaction terms, temporal trends, geospatial features) for more predictive power.  
  - Explore feature scaling and hyperparameter tuning to tighten MAE toward the 15–20% target band.  
  - Monitor for data drift and plan regular retraining to keep performance stable.

- **Practical takeaway:**  
  The current model already delivers a meaningful predictive signal. For production use, aim for incremental improvements that close the MAE gap while keeping deployment and update processes simple (see **Model versioning** above).