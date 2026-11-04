from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.config import DEFAULT_THRESHOLD
from src.explainability import shap_factors
from src.explanation import human_explanation
from src.inference import predict_default
from src.model_loader import load_model
from src.schemas import Applicant, HealthResponse, Prediction

logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting credit-risk prediction API")
    app.state.model = load_model()
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
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        features = pd.DataFrame([applicant.model_dump()])
        result = predict_default(model, features, DEFAULT_THRESHOLD).iloc[0]
        probability = float(result["default_probability"])
        predicted_default = int(result["predicted_default"])
        factors = shap_factors(model, features)
        explanation, source = human_explanation(
            probability,
            DEFAULT_THRESHOLD,
            predicted_default,
            factors,
        )
    except ValueError as error:
        logger.warning("Invalid prediction input: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None

    return Prediction(
        default_probability=probability,
        predicted_default=predicted_default,
        threshold=DEFAULT_THRESHOLD,
        shap_factors=factors,
        explanation=explanation,
        explanation_source=source,
    )
