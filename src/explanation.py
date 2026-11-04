from __future__ import annotations

import logging
import os
from json import dumps, loads
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from src.schemas import ShapFactor

logger = logging.getLogger("credit_risk_api")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))


def _fallback(
    probability: float,
    threshold: float,
    factors: list[ShapFactor],
) -> str:
    decision = "higher" if probability >= threshold else "lower"
    factor_text = ", ".join(
        f"{factor.feature} ({factor.impact})" for factor in factors[:3]
    )
    return (
        f"The model estimates a {probability:.1%} default probability, "
        f"which is above the {threshold:.0%} threshold. The applicant is "
        f"classified as {decision} risk. The strongest model factors were "
        f"{factor_text}. This explains model behavior and is not a definitive "
        "financial decision."
    )


def human_explanation(
    probability: float,
    threshold: float,
    predicted_default: int,
    factors: list[ShapFactor],
) -> tuple[str, str]:
    fallback = _fallback(probability, threshold, factors)
    prompt = (
        "Write one concise professional explanation of this machine-learning "
        "prediction. Use only supplied facts. Do not invent details, causes, "
        "recommendations, or numbers. Preserve each factor's impact. Do not "
        "change the prediction. State that this describes model behavior and "
        "is not a definitive financial decision.\n\n"
        f"Default probability: {probability:.4f}\n"
        f"Decision threshold: {threshold:.2f}\n"
        f"Predicted default: {predicted_default}\n"
        f"SHAP factors: {[factor.model_dump() for factor in factors]}\n"
    )
    request = Request(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        data=dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            payload = loads(response.read().decode("utf-8"))
        explanation = payload.get("response")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Ollama returned an empty explanation.")
        return explanation.strip(), "ollama"
    except (OSError, URLError, ValueError, TimeoutError) as error:
        logger.warning("Ollama explanation unavailable: %s", error)
        return fallback, "deterministic_fallback"
