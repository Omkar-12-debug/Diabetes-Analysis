# Diabetes Prediction MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for diabetes prediction, featuring reproducible environments, pipeline orchestration, model registry, artifact versioning, and continuous serving.

## Quick Start (One-Command Setup)

Ensure you have [uv](https://github.com/astral-sh/uv) installed (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows or `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux/macOS).

Then, run the one-command setup:

```bash
uv sync
```

This will automatically create a local virtual environment (`.venv`), resolve Python (>=3.11), and deterministically install all locked dependencies from `uv.lock`.

### Environment Configuration

Copy the example environment file to configure your local setup:

```bash
cp .env.example .env
```

### Running Tests

Run the test suite using `uv`:

```bash
uv run pytest
```

### Starting the API Server

Launch the FastAPI application:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

## Project Structure

```text
MLOPS_Diabetes/
├── .github/workflows/    # CI/CD automated pipeline workflows
├── app/                  # FastAPI inference serving
│   ├── routes/           # API route handlers
│   ├── services/         # Inference logic & business logic
│   ├── main.py           # Application entrypoint
│   └── schemas.py        # Pydantic request/response schemas
├── dags/                 # Orchestration DAGs (Airflow)
├── data/
│   ├── raw/              # Raw ingested datasets (tracked via DVC)
│   └── processed/        # Cleaned and engineered datasets (tracked via DVC)
├── frontend/             # User interface components
├── infra/                # Infrastructure as Code (Docker, Terraform, etc.)
├── models/               # Model binaries and serialized artifacts
├── monitoring/           # Data and model drift monitoring scripts & dashboards
├── reports/              # Generated analysis figures, metrics, and logs
├── src/                  # Core ML pipeline modules
│   ├── data/             # Ingestion & loading utilities
│   ├── preprocessing/    # Data cleaning & imputation pipelines
│   ├── features/         # Feature extraction & engineering logic
│   ├── training/         # Model training & hyperparameter tuning
│   ├── evaluation/       # Validation metrics & model evaluation
│   └── monitoring/       # Drift detection & telemetry logic
├── tests/                # Automated unit and integration tests
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── pyproject.toml        # Project configuration & dependency definitions
├── README.md             # Project documentation
└── uv.lock               # Strictly pinned lockfile
```
