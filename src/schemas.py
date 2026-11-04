from pydantic import BaseModel, ConfigDict, Field


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


class ShapFactor(BaseModel):
    feature: str
    impact: str
    shap_value: float


class Prediction(BaseModel):
    default_probability: float
    predicted_default: int
    threshold: float
    shap_factors: list[ShapFactor]
    explanation: str
    explanation_source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
