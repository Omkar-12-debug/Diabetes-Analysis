<div align="center">

# 🩺 DiabetesRisk AI

### An Explainable Machine Learning & MLOps Platform for Personalized Diabetes Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-2.10-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com)
[![Tests](https://img.shields.io/badge/Tests-145%20Passed-brightgreen?style=for-the-badge)](https://pytest.org)

<br/>

**A production-ready clinical risk assessment platform that evaluates individualized diabetes risk using calibrated gradient boosting, interprets key metabolic drivers with TreeSHAP, generates actionable counterfactual pathways via DiCE, and operates within an enterprise MLOps lifecycle.**

<br/>

</div>

---

## 1. Problem Statement

Diabetes mellitus is one of the fastest-growing chronic health conditions worldwide[cite: 13]. Early detection and risk stratification are essential for timely intervention, lifestyle modifications, and targeted clinical screenings. However, practical risk assessment involves complex interactions across physiological, metabolic, and demographic variables:

1. **Multi-Factor Interaction**: Fasting blood glucose and HbA1c levels interact heavily with age, body mass index (BMI), hypertension, and cardiovascular history[cite: 13, 14]. Simple threshold-based heuristics often overlook borderline or compounding risks[cite: 14].
2. **The Black-Box Dilemma**: Standard machine learning models generate isolated binary labels (diabetic vs. non-diabetic) without explaining *why* an individual is categorized at high risk[cite: 13]. Without interpretable feature attributions, patients and clinicians lack the clarity needed to trust algorithmic evaluations[cite: 13].
3. **Absence of Actionable Pathways**: Knowing that a risk score is high does not inform an individual what changes would most effectively reduce that risk[cite: 13, 14]. An effective platform must allow users to simulate personalized, realistic adjustments to modifiable factors while holding fixed demographic attributes constant[cite: 13, 14].

This project implements an end-to-end clinical machine learning framework that delivers calibrated risk scores, feature explanations, and counterfactual simulation through an accessible web interface and a resilient production infrastructure[cite: 13].

---

## 2. Proposed Solution

DiabetesRisk AI provides an integrated system that transforms clinical parameters into personalized, transparent risk assessments and simulated mitigation pathways[cite: 13]:

- **Calibrated Risk Stratification**: Computes continuous, well-calibrated probabilities mapped to clear risk tiers—**Low (<30)**, **Moderate (30–70)**, and **High (>70)**—preventing arbitrary binary classifications[cite: 13].
- **Explainable Predictions (TreeSHAP)**: Identifies the exact biometric drivers behind every individual score, revealing whether elevated glucose, BMI, or blood pressure contributed most to the outcome[cite: 13].
- **Actionable Counterfactual Analysis (DiCE)**: An interactive "What-If" simulator that computes the minimal, realistic lifestyle modifications (e.g., reducing BMI from 32.5 to 27.8 or adjusting glycemic targets) required to bring a high-risk profile down to a safe baseline, while strictly locking non-modifiable attributes like age and gender[cite: 13, 14].
- **Patient Assessment History**: Tracks chronological risk trajectories across multiple evaluations to visualize personal progress over time[cite: 13, 14].
- **Production MLOps Lifecycle**: Built on an enterprise foundation—automated data validation, Airflow DAG orchestration, MLflow experiment tracking, Prometheus/Grafana observability, Evidently AI drift detection, and AWS deployment automation[cite: 13].

---

## 3. Machine Learning Methodology

### 3.1 Data Preparation & Preprocessing
The benchmark dataset consists of **100,000 patient records** evaluated across clinical and metabolic markers[cite: 13, 14]:
- **Validation**: Strict schema verification enforcing physiological boundaries (e.g., BMI between 10.0 and 70.0, HbA1c between 2.0% and 20.0%, Blood Glucose between 20.0 and 600.0 mg/dL)[cite: 13].
- **Exact Deduplication**: Identifies and removes 3,854 duplicate entries, resulting in 96,146 unique patient records[cite: 13].
- **Stratified Partitioning**: Partitioned into **70% Training** ($N = 67,302$), **15% Validation** ($N = 14,422$), and **15% Test** ($N = 14,422$) splits, preserving the positive class ratio across all sets[cite: 13].
- **Feature Engineering**: Derives key clinical interactions:
  - `cardiometabolic_risk`: Composite burden combining hypertension, heart disease, and obesity status[cite: 13].
  - `glucose_hba1c_interaction`: Product interaction capturing metabolic severity[cite: 13].
  - `age_bmi_interaction`: Interaction modeling progressive metabolic slowdown with age[cite: 13].
  - `bmi_category` & `age_group`: WHO standard clinical categorizations[cite: 13].
- **Imbalance Handling**: Mitigates the 8.5% positive class imbalance using cost-sensitive learning (`scale_pos_weight`) and optimizing for Precision-Recall AUC (PR-AUC) rather than raw accuracy[cite: 13, 14].
- **Probability Calibration**: Uses Sigmoid/Platt calibration to align raw tree scores with true observed empirical frequencies, achieving a clinical Brier score of **0.0561**[cite: 13].

### 3.2 Model Hierarchy & Benchmarking Results

Across extensive evaluation on the held-out test cohort ($N = 14,422$), the candidate models were compared against an interpretable baseline[cite: 13]:

| Model Family | Role | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Brier Score | Status |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.6124 | 0.8841 | 64.21% | 52.18% | 0.5757 | 0.0812 | Evaluated |
| **Gaussian Naive Bayes** | Benchmark | 0.5891 | 0.8654 | 72.45% | 41.30% | 0.5260 | 0.1142 | Evaluated |
| **Decision Tree** | Benchmark | 0.6934 | 0.8372 | 74.12% | 68.45% | 0.7117 | 0.0784 | Evaluated |
| **Random Forest** | Candidate 1 | 0.8645 | 0.9692 | 84.32% | 58.74% | 0.6923 | 0.0594 | Evaluated |
| **XGBoost (Calibrated)** | Candidate 2 | **0.8860** | **0.9779** | **90.57%** | **48.65%** | **0.6330** | **0.0561** | **Champion** |

*The Calibrated XGBoost model was selected as the production `@champion` model in the MLflow Model Registry due to its superior PR-AUC (0.8860) and clinical recall (90.57%), ensuring high sensitivity for screening risk[cite: 13].*

---

## 4. System Architecture

### 4.1 End-to-End System Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION LAYER                            │
│                                                                             │
│  1. Clinician / Patient opens React Dashboard                               │
│  2. Inputs health metrics (Age, Gender, BMI, HbA1c, Glucose, etc.)          │
│  3. Submits profile → React Query dispatches asynchronous POST to FastAPI   │
├─────────────────────────────────────────────────────────────────────────────┤
│                          API & ML INFERENCE LAYER                           │
│                                                                             │
│  4. FastAPI validates request schema against physiological bounds           │
│  5. Preprocessor transforms inputs & synthesizes engineered features        │
│  6. XGBoost Champion Model generates calibrated risk score & tier           │
│  7. TreeSHAP computes patient-specific biomarker attribution weights        │
│  8. DiCE engine solves optimization problem for actionable counterfactuals   │
│  9. Assessment & telemetry logged to PostgreSQL / SQLite database           │
├─────────────────────────────────────────────────────────────────────────────┤
│                         OBSERVABILITY & MONITORING                          │
│                                                                             │
│  10. Prometheus scrapes /metrics (latency, request counts, score tiers)     │
│  11. Grafana provides real-time operational dashboard visualization         │
│  12. Evidently AI inspects input batches for statistical feature drift      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline Architecture

```
                ┌─────────────────────────┐
                │      React Frontend     │
                │   (Vite + TypeScript)   │
                └────────────┬────────────┘
                             │ HTTP REST
                             ▼
                ┌─────────────────────────┐
                │     FastAPI Backend     │
                │      (app/main.py)      │
                └────────────┬────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Ingestion Service│ │ Predictor Service│ │ Explain Service  │
│   (src/data/)    │ │(models/*.joblib) │ │  (SHAP + DiCE)   │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Apache Airflow  │ │ MLflow Registry  │ │  Database Layer  │
│  Orchestration   │ │ @champion Model  │ │  PostgreSQL/SQL  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 5. Repository Structure

```
Diabetes-Analysis/
├── app/                        # FastAPI Service & Application Layer
│   ├── main.py                 # Application entrypoint & middleware
│   ├── schemas.py              # Pydantic clinical validation schemas
│   ├── db/                     # Database sessions & models
│   ├── monitoring/             # Prometheus & CloudWatch telemetry
│   ├── routes/                 # REST routes (/predict, /explain, /history)
│   └── services/               # Business logic (Predictor, SHAP, DiCE)
│
├── frontend/                   # Modern React Clinical Dashboard
│   ├── src/                    # TypeScript source (Pages, Components)
│   ├── package.json            # Frontend dependencies
│   └── vite.config.ts          # Vite build configuration
│
├── src/                        # Machine Learning Core Pipeline
│   ├── data/                   # Ingestion & deterministic generators
│   ├── preprocessing/          # Leakage-safe transformers & scalers
│   ├── features/               # Clinical feature engineering & SHAP
│   ├── training/               # Baseline, RF, and XGBoost training
│   ├── evaluation/             # PR-AUC, calibration & fairness audits
│   └── monitoring/             # Evidently AI drift & delayed-label tracking
│
├── dags/                       # Workflow Orchestration
│   └── diabetes_ml_pipeline.py # 10-stage Apache Airflow TaskFlow DAG
│
├── infra/                      # Cloud & Deployment Infrastructure
│   └── cloud/                  # AWS SageMaker deploy & cleanup automation
│
├── monitoring/                 # Observability Stack
│   ├── prometheus/             # Scrape configurations
│   └── grafana/                # Pre-provisioned dashboards & datasources
│
├── reports/                    # Clinical Documentation & Audits
│   ├── model_card.md           # Formal MLflow Model Card
│   ├── data_card.md            # Dataset provenance & audit findings
│   └── performance_report.md   # Operational & calibration metrics
│
├── tests/                      # Comprehensive Automated Pytest Suite
│   ├── test_api.py             # REST API endpoint tests
│   ├── test_drift_monitoring.py# Evidently AI shift tests
│   ├── test_cloud_deployment.py# AWS SageMaker simulation tests
│   └── test_metrics.py         # Prometheus metrics validation
│
├── docker-compose.yml          # Multi-service container orchestration
├── Dockerfile                  # Multi-stage production container build
├── pyproject.toml              # Locked project dependencies
└── uv.lock                     # Astral uv reproducible lockfile
```

---

## 6. Quickstart Guide

### Option A: Local Development via `uv`

Ensure Python 3.11+ and [Astral `uv`](https://github.com/astral-sh/uv) are installed:

```bash
# 1. Clone the repository
git clone https://github.com/Omkar-12-debug/Diabetes-Analysis.git
cd Diabetes-Analysis

# 2. Sync locked environment dependencies
uv sync

# 3. Generate data splits and train baseline models
uv run python -m src.preprocessing.pipeline
uv run python -m src.training.train_all

# 4. Start the FastAPI backend server
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a separate terminal, launch the frontend:

```bash
cd frontend
npm install
npm run dev
```

Access the UI at http://localhost:5173 and the interactive Swagger API docs at http://127.0.0.1:8000/docs.

### Option B: One-Command Docker Compose

To spin up the entire multi-service stack (FastAPI Backend, React Dashboard, PostgreSQL, Prometheus, and Grafana):

```bash
docker compose up --build -d
```

| Service | Endpoint | Description |
|---|---|---|
| **Clinical Dashboard** | http://localhost:3000 | Interactive React web interface |
| **FastAPI REST API** | http://localhost:8000 | Inference engine & Swagger docs |
| **Prometheus** | http://localhost:9090 | Time-series metrics engine |
| **Grafana** | http://localhost:3001 | Pre-configured observability dashboards |

---

## 7. Monitoring, Drift Detection & Observability

### 7.1 Prometheus Metrics Catalog
The FastAPI service exposes real-time application metrics at `GET /metrics`:

| Metric | Type | Description |
|---|---|---|
| `prediction_requests_total` | Counter | Total predictions stratified by `model_version` and `risk_category` |
| `prediction_errors_total` | Counter | Total inference errors by `endpoint` and `error_type` |
| `prediction_latency_seconds` | Histogram | Request latency distribution across standard latency buckets |
| `prediction_risk_score` | Histogram | Continuous distribution of calibrated risk scores (0–100) |
| `active_requests_in_flight` | Gauge | Real-time concurrent requests currently being processed |

### 7.2 Evidently AI Drift Detection
Data drift detection is performed by evaluating incoming patient feature distributions against the baseline reference cohort using Kolmogorov-Smirnov (KS) tests for numerical features and Chi-Square tests for categorical features:
- **In-Distribution Baseline**: Continuous evaluation confirms statistical stability ($p > 0.05$, zero drift flags).
- **Controlled Clinical Shift Simulation**: When exposed to shifted populations (e.g., an older cohort with elevated fasting glucose and HbA1c), the drift engine detects distribution shifts ($p < 0.01$), flags a 44.44% drift share, and updates the status at `GET /monitoring/drift/report`.

---

## 8. Cloud Architecture & Cost Management (AWS)

The platform includes cloud deployment automation designed for AWS, with built-in controls to prevent unnecessary cloud expenditure:
- **AWS S3 Artifact Synchronization (`src/cloud/s3_sync.py`)**: Manages seamless synchronization of dataset splits, serialized model artifacts, and evaluation reports with Amazon S3 storage buckets.
- **SageMaker Deployment Pipeline (`infra/cloud/sagemaker_deploy.py`)**: Packages the registered `@champion` XGBoost model into a SageMaker-compatible container tarball, sets up endpoint configurations, and provisions a managed real-time inference endpoint. Includes a full `--dry-run` simulation mode to test deployment workflows offline.
- **CloudWatch Telemetry (`app/monitoring/cloudwatch.py`)**: Streams inference latency and prediction distributions directly to AWS CloudWatch Logs and custom metrics namespaces.
- **Automated Cost-Control Cleanup (`infra/cloud/sagemaker_cleanup.py`)**: Systematically tears down the active SageMaker endpoint, deletes the endpoint configuration, and removes model resources to guarantee zero ongoing billing following demonstrations:

```bash
uv run python -m infra.cloud.sagemaker_cleanup --endpoint-name diabetes-risk-endpoint
```

---

## 9. Testing & Quality Assurance

The repository enforces strict continuous integration standards via GitHub Actions with 145 automated tests across all pipeline layers:

```bash
# Run the complete test suite locally
uv run pytest -v
```

- **Validation & Pipeline Tests (`test_validation.py`, `test_preprocessing.py`)**: Enforce physiological boundary checks, deterministic train/val/test splits, and data transformations.
- **Model Evaluation Tests (`test_evaluation.py`, `test_training.py`)**: Verify PR-AUC computations, probability calibration bounds, and demographic fairness parity.
- **XAI & Explainability Tests (`test_xai.py`)**: Assert local SHAP attribution additivity and verify that DiCE counterfactuals strictly preserve non-modifiable demographic features.
- **API & Observability Tests (`test_api.py`, `test_metrics.py`, `test_drift_monitoring.py`)**: Verify all REST endpoints, database logging, Prometheus metric counters, and Evidently drift simulations.
- **Cloud Deployment Contracts (`test_cloud_deployment.py`)**: Validates SageMaker artifact packaging, deployment dry-runs, and teardown logic.

---

## 10. Ethical Boundaries & Clinical Disclaimer

### IMPORTANT CLINICAL NOTICE
**DiabetesRisk AI is an academic machine learning research system and clinical decision-support screening aid. It is not an autonomous medical device, diagnostic system, or treatment prescription tool.**

- **Screening Aid Only:** The predicted risk score represents an empirical estimation based on statistical associations in observational data; it does not constitute a formal clinical diagnosis.
- **Non-Causal Counterfactuals:** What-If counterfactual recommendations generated by DiCE illustrate model sensitivity and statistical feature boundaries. They should not be interpreted as guaranteed medical advice or prescriptive health interventions.
- **Clinical Supervision:** All diagnostic assessments, treatment plans, and therapeutic changes must be performed under the direct supervision of a licensed healthcare professional.
