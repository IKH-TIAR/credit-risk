from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent


# Data Directory
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# MLflow configuration
MLFLOW_TRACKING_URI = (
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = "credit-risk-default-prediction"

# Target variable
TARGET_COLUMN = "loan_status"

# Random state for reproducibility
RANDOM_STATE = 42