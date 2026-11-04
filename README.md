# Credit Risk Default Prediction

Machine-learning project for predicting loan default risk. The project includes
data exploration, preprocessing, Random Forest tuning, SHAP explainability, and
a FastAPI prediction service.

## Run the API with Docker

Requirements:

- Docker Desktop with the Linux engine running
- The locally generated MLflow model artifact under `mlruns/`

Start the API:

```powershell
docker compose up --build -d
```

The API is available at:

- Swagger UI: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Stop and remove the container:

```powershell
docker compose down
```

## Prediction request

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{
    "person_age": 22,
    "person_income": 59000,
    "person_emp_length": 4,
    "loan_amnt": 35000,
    "loan_percent_income": 0.59,
    "cb_person_cred_hist_length": 3,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "Y"
  }'
```

The response contains:

```json
{
  "default_probability": 0.84,
  "predicted_default": 1,
  "threshold": 0.35
}
```

The API uses the final tuned Random Forest and the validation-selected
classification threshold of `0.35`. Applicant values are not written to logs.

## Run locally without Docker

```powershell
.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

## Project structure

```text
data/          Raw and processed datasets
notebooks/     EDA, training, tuning, and explainability notebooks
src/           Preprocessing, inference, and API code
Dockerfile     API container image
docker-compose.yml
```
