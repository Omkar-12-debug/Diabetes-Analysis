# 🩺 DiabetesRisk AI — Presentation & Demonstration Guide

This guide contains everything you need to deliver a flawless, high-scoring presentation of the **Diabetes Risk MLOps Platform**.

---

## 🚀 Quick Launch (Before You Present)

1. Open **Docker Desktop** on your laptop (from your Desktop or Start Menu).
2. Double-click `launch_stack.bat` in the project root folder (or run `docker compose up -d`).
3. Your browser will automatically open with all necessary dashboards:

| Service | Local URL | Credentials | Purpose in Demo |
| :--- | :--- | :--- | :--- |
| **React Clinical Webpage** | [http://localhost:5173](http://localhost:5173) | *None* | Main demonstration UI with sliders, presets, SHAP, and What-If |
| **Grafana Observability** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` | Real-time p50/p95 latency, throughput, and risk distribution telemetry |
| **Prometheus Target Status** | [http://localhost:9090/targets](http://localhost:9090/targets) | *None* | Verifies the `api:8000/metrics` target is UP and actively scraped |
| **FastAPI Swagger Playground** | [http://localhost:8000/docs](http://localhost:8000/docs) | *None* | Interactive REST endpoints (`/predict`, `/explain`, `/what-if`, `/metrics`) |
| **MLflow Model Registry** | [http://localhost:5000](http://localhost:5000) | *None* | Model lineage, version history, and `@champion` alias assignment |

---

## 🎬 5-Minute Live Demo Script (Step-by-Step)

### Step 1: Introduction & Landing Page (1 Minute)
* Open [http://localhost:5173](http://localhost:5173).
* Point out the **Model Governance Pill** in the navbar: `Champion Model: XGBoost @champion v2 (PR-AUC: 0.886 · Recall: 90.6%)`.
* Point out the **Quick-Launch Links** (`Grafana`, `Prometheus`, `API Docs`) in the navbar.
* *What to Say*: *"Our project is DiaGuard AI, an explainable clinical decision support platform. Instead of treating machine learning as an opaque black box, we deliver calibrated risk scores, explain biometric drivers with TreeSHAP, and offer actionable lifestyle mitigation pathways with DiCE counterfactuals—all monitored by an enterprise MLOps telemetry stack."*

### Step 2: Patient Assessment & Presets (1 Minute)
* Click **Check Risk** (or the blue button).
* Demonstrate the **Preset Buttons**:
  1. Click **"Healthy Adult"** $\rightarrow$ Click **"Calculate Risk Score"**.
     * Show the result: **Low Risk Tier (<30)**.
  2. Click **Back** $\rightarrow$ Click **"Elevated Risk"** (or Borderline) $\rightarrow$ Click **"Calculate Risk Score"**.
     * Show the result: **High Risk Tier (>70)**.
* *What to Say*: *"We validate all clinical inputs against physiological limits. Raw tree outputs are Platt-calibrated into continuous, true empirical probabilities rather than arbitrary threshold flips."*

### Step 3: Explainability — TreeSHAP (1 Minute)
* On the Results page, scroll down to **"Factors Increasing Your Risk"** and **"Factors Working in Your Favor"**.
* Point out the specific numbers (e.g., Blood Glucose 195 mg/dL and HbA1c 7.8% driving the risk upwards, while younger age works in their favor).
* *What to Say*: *"Physicians and patients cannot trust unexplained labels. Using local Shapley values, our model breaks down exactly which metabolic biomarkers drove this patient's risk higher, providing clinical justification."*

### Step 4: Actionable What-If Counterfactual Simulator (1 Minute)
* Click **"Open Simulator"** (or the **What-If Simulator** tab).
* Drag the **BMI slider down** (e.g. from 34.2 to 26.0) and **Blood Glucose slider down** (e.g. from 195 to 115).
* Watch the projected risk score drop from **High** to **Low**.
* *What to Say*: *"Telling a patient their risk is high isn't enough—they need actionable steps. Our simulator locks immutable traits like age and biological sex, and tests feasible lifestyle modifications to show how targeted reductions in BMI and glucose can safely transition them into the healthy tier."*

### Step 5: MLOps Telemetry & Grafana Spikes (1 Minute)
* Click the **Grafana** link in the navbar (or switch to `http://localhost:3000`).
* Set the time filter to **Last 5 minutes** with **Auto-refresh: 5s**.
* Show the 9 panels:
  * **HTTP Request Rate (RPS)**: Shows live spikes from the predictions you just executed.
  * **Latency Distributions (p50 / p95)**: Sub-50ms inference times.
  * **Risk Category Distribution Donut**: Shows the breakdown of Low vs. High risk requests logged.
* *What to Say*: *"Every prediction event streams live telemetry into Prometheus and logs to our PostgreSQL audit database. This allows us to track latency, catch concept drift, and audit model performance continuously in production."*

---

## 💡 Top Professor & Evaluator Questions (Cheat Sheet)

#### Q1: "Why did you prioritize PR-AUC and Recall instead of standard Accuracy?"
> **Answer**: *"The dataset has a severe 10.3:1 class imbalance (~8.5% positive diabetic cases). A trivial model predicting 'Healthy' for every single person would achieve 91.5% accuracy, but 0% recall, fatally missing all high-risk patients. For clinical screening, high sensitivity/recall (90.57%) and Precision-Recall AUC (0.8860) ensure we minimize false negatives."*

#### Q2: "Which models were evaluated, and why is XGBoost the Champion?"
> **Answer**: *"We evaluated 5 models: Logistic Regression baseline (PR-AUC 0.612), Gaussian Naive Bayes (0.589), Decision Trees (0.693), Random Forest (0.865), and XGBoost. Tuned with cost-sensitive `scale_pos_weight` and Sigmoid probability calibration, XGBoost achieved the best PR-AUC (0.8860), highest recall (90.57%), and the lowest calibration error (Brier score: 0.0561)."*

#### Q3: "What prevents data leakage during preprocessing?"
> **Answer**: *"All scalers and one-hot encoders are fitted strictly on the 70% training split. The validation and test splits are transformed using the train-fitted transformer (`models/preprocessor.joblib`), ensuring zero test distribution information leaks into feature scaling or parameter selection."*

#### Q4: "How does the system handle an offline backend or database disruption?"
> **Answer**: *"The frontend client implements defensive programming with smart mock fallbacks. If network latency spikes or a container restarts, the UI gracefully falls back to deterministic in-memory estimation without throwing unhandled exceptions or breaking the user experience."*

---

## 🛑 Teardown Command
When your presentation is done, stop the containers safely:
```bash
double-click stop_stack.bat
# or in terminal:
docker compose stop
```

