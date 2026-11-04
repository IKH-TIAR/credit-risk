from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_THRESHOLD

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _prepare_features(applicants: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw applicant records for model inference."""
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in applicants.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required applicant columns: "
            + ", ".join(missing_columns)
        )

    features = applicants.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        try:
            features[column] = pd.to_numeric(features[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Numeric feature '{column}' contains a non-numeric value."
            ) from error
        features[column] = features[column].astype("float64")

    return features


def predict_default(
    model: Pipeline,
    applicants: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Return default probabilities and thresholded predictions.

    The model must be the fitted preprocessing-plus-classifier pipeline logged
    from the final training workflow.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    features = _prepare_features(applicants)
    probabilities = model.predict_proba(features)[:, 1]

    return pd.DataFrame(
        {
            "default_probability": probabilities,
            "predicted_default": (probabilities >= threshold).astype("int8"),
        },
        index=applicants.index,
    )


def model_feature_columns() -> tuple[str, ...]:
    """Return the raw feature columns expected by the final model."""
    return tuple(FEATURE_COLUMNS)
