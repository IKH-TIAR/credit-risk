from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.schemas import ShapFactor


def _display_feature_name(feature: str) -> str:
    display_name = feature.split("__", 1)[-1].replace("_", " ")
    return display_name.replace("amnt", "amount")


def shap_factors(
    model: Pipeline,
    features: pd.DataFrame,
    limit: int = 5,
) -> list[ShapFactor]:
    import shap

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    processed_features = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out()
    values = shap.TreeExplainer(classifier).shap_values(processed_features)

    if isinstance(values, list):
        default_values = values[1][0]
    else:
        default_values = values[0]
        if default_values.ndim == 2:
            default_values = default_values[:, 1]

    ranked = sorted(
        zip(feature_names, default_values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:limit]
    return [
        ShapFactor(
            feature=_display_feature_name(feature),
            impact="increases risk" if value > 0 else "reduces risk",
            shap_value=float(value),
        )
        for feature, value in ranked
    ]
