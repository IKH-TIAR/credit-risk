from __future__ import annotations

import logging
import os

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from src.config import MLFLOW_TRACKING_URI

logger = logging.getLogger("credit_risk_api")

MODEL_RUN_NAME = "random-forest-final-tuned"
MODEL_ARTIFACT_PATH = "random-forest-final-tuned-model"
MODEL_URI = os.getenv("CREDIT_RISK_MODEL_URI")


def load_model() -> Pipeline:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = MODEL_URI

    if model_uri is None:
        runs = mlflow.search_runs(
            experiment_names=["credit-risk-default-prediction"],
            filter_string=f"tags.mlflow.runName = '{MODEL_RUN_NAME}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(
                "No final tuned model run found in MLflow. "
                "Set CREDIT_RISK_MODEL_URI or train the final model first."
            )
        model_uri = f"runs:/{runs.iloc[0]['run_id']}/{MODEL_ARTIFACT_PATH}"

    logger.info("Loading model from configured MLflow URI")
    model = mlflow.sklearn.load_model(model_uri)
    if not isinstance(model, Pipeline):
        raise TypeError("Configured MLflow model is not a scikit-learn Pipeline.")
    logger.info("Model loaded successfully")
    return model
