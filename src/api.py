from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_THRESHOLD,
    MLFLOW_TRACKING_URI,
)
from src.inference import predict_default

logger = logging.getLogger("credit_risk_api")

MODEL_RUN_NAME = "random-forest-final-tuned"
MODEL_ARTIFACT_PATH = "random-forest-final-tuned-model"
MODEL_URI = os.getenv("CREDIT_RISK_MODEL_URI")


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_age: float | None = Field(default=None, ge=0)
    person_income: float | None = Field(default=None, ge=0)
    person_emp_length: float | None = Field(default=None, ge=0)
    loan_amnt: float | None = Field(default=None, ge=0)
    loan_percent_income: float | None = Field(default=None, ge=0)
    cb_person_cred_hist_length: float | None = Field(default=None, ge=0)
    person_home_ownership: str
    loan_intent: str
    cb_person_default_on_file: str


class Prediction(BaseModel):
    default_probability: float
    predicted_default: int
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _load_model() -> Pipeline:
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
        model_uri = (
            f"runs:/{runs.iloc[0]['run_id']}/{MODEL_ARTIFACT_PATH}"
        )

    logger.info("Loading model from configured MLflow URI")
    model = mlflow.sklearn.load_model(model_uri)
    if not isinstance(model, Pipeline):
        raise TypeError("Configured MLflow model is not a scikit-learn Pipeline.")
    logger.info("Model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting credit-risk prediction API")
    app.state.model = _load_model()
    yield
    app.state.model = None
    logger.info("Stopped credit-risk prediction API")


app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request failure method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Request completed method=%s path=%s status=%s duration_ms=%.2f "
        "request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant, request: Request) -> Prediction:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:
        features = pd.DataFrame([applicant.model_dump()])
        result = predict_default(model, features, DEFAULT_THRESHOLD).iloc[0]
    except ValueError as error:
        logger.warning("Invalid prediction input: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from None

    return Prediction(
        default_probability=float(result["default_probability"]),
        predicted_default=int(result["predicted_default"]),
        threshold=DEFAULT_THRESHOLD,
    )
