# Data Card: Diabetes Prediction Dataset

A standardized clinical dataset card conforming to the guidelines of **Gebru et al. (2021)** (*"Datasheets for Datasets"*, Communications of the ACM).

---

## 1. Dataset Motivation & Provenance

- **Dataset Identifier:** `Diabetes Prediction Dataset`
- **Canonical Storage:** `data/raw/diabetes_prediction_dataset.csv`
- **Volume:** 100,000 raw observation records
- **Format:** Comma-Separated Values (UTF-8, RFC 4180)
- **Domain:** Outpatient Preventive Medicine & Chronic Metabolic Disease Screening
- **Curator / Origin:** Aggregation of electronic medical health records (EHR) and standardized metabolic survey panels from diverse ambulatory outpatient networks.
- **License:** Open Access / Research Use (CC BY 4.0 International)
- **Primary Clinical Purpose:** Training and validating machine learning models for early risk detection of type 2 diabetes mellitus (T2DM) to assist primary care screening.

---

## 2. Dataset Composition & Attribute Dictionary

The raw dataset contains 9 columns (8 predictive features and 1 ground-truth binary target).

| Feature Name | Clinical Description | Data Type | Physical Range | Permitted Validation Bounds | Null Count |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `gender` | Biological sex or administrative gender identity | Categorical | `Female`, `Male`, `Other` | Permitted set of 3 classes | 0 (0.0%) |
| `age` | Patient chronological age at examination | Numerical (float) | 0.08 to 80.0 years | $[0.0, 120.0]$ | 0 (0.0%) |
| `hypertension` | Clinical diagnosis of persistent essential hypertension | Binary (int) | 0 (No), 1 (Yes) | $\{0, 1\}$ | 0 (0.0%) |
| `heart_disease` | Documented coronary artery or cardiovascular disease | Binary (int) | 0 (No), 1 (Yes) | $\{0, 1\}$ | 0 (0.0%) |
| `smoking_history` | Self-reported or documented tobacco exposure | Categorical | 6 categories | `never`, `No Info`, `current`, `former`, `ever`, `not current` | 0 (0.0%) |
| `bmi` | Body Mass Index ($\text{weight } [\text{kg}] / \text{height } [\text{m}]^2$) | Numerical (float) | 10.01 to 95.69 $kg/m^2$ | $[10.0, 70.0]$ (clipped $>70$) | 0 (0.0%) |
| `HbA1c_level` | Glycated hemoglobin fraction | Numerical (float) | 3.5% to 9.0% | $[2.0, 20.0]$ | 0 (0.0%) |
| `blood_glucose_level` | Instantaneous blood glucose concentration | Numerical (int) | 80 to 300 $mg/dL$ | $[20, 600]$ | 0 (0.0%) |
| `diabetes` | Confirmed medical diagnosis of diabetes (Target) | Binary (int) | 0 (Non-diabetic), 1 (Diabetic) | $\{0, 1\}$ | 0 (0.0%) |

---

## 3. Class Distribution & Severe Imbalance

### Raw Class Prevalence ($N = 100,000$)
| Class Label | Clinical Status | Count ($N$) | Percentage (%) |
| :---: | :--- | :---: | :---: |
| **0** | Non-Diabetic | 91,500 | **91.50%** |
| **1** | Confirmed Diabetic | 8,500 | **8.50%** |
| **Total** | Full Cohort | 100,000 | **100.00%** |

### Mathematical Implication of Imbalance
- **Negative-to-Positive Imbalance Ratio:** $\frac{91,500}{8,500} = 10.76 : 1$.
- **The Naive Classifier Paradox:** A trivial dummy classifier that unconditionally predicts class 0 for every patient achieves **91.50% classification accuracy** while missing **100% of diabetic patients** ($\text{Recall} = 0.0\%$).
- **Primary Optimization Metric:** Raw accuracy is clinically dangerous and unacceptable. Optimization and model ranking are governed by **PR-AUC (Precision-Recall Area Under Curve)**, **Recall / Sensitivity**, and **Brier Score Calibration**.

---

## 4. Rigorous Data Quality Audit & Anomaly Findings

An automated statistical profiling audit on the raw dataset identified three critical phenomena:

### 4.1 Exact Duplicate Records ($N = 3,854$)
- **Audit Finding:** Exactly **3,854 duplicate rows** (3.854% of the cohort) exist with identical values across all 9 features including the target.
- **Root Cause Analysis:** Typical in EHR extractions where repeated administrative entries or multi-visit records with identical vitals are exported without unique visit keys.
- **Handling Strategy:** All 3,854 exact duplicate rows are purged during preprocessing prior to dataset partitioning. This prevents **cross-fold data leakage** between training and evaluation splits, reducing the clean dataset size to **96,146 unique rows**.

### 4.2 Sub-Year / Infant Records ($N = 911$)
- **Audit Finding:** **911 records** (0.911% of the cohort) possess an `age < 1.0` year (minimum age = 0.08 years / ~1 month).
- **Clinical Evaluation:**
  - Standard laboratory Glycated Hemoglobin ($\text{HbA1c}$) interpretation guidelines (such as the ADA standards) are established for adults and children $>1$ year. In neonates, fetal hemoglobin ($\text{HbF}$) can confound standard laboratory chromatography assays.
  - Furthermore, BMI calculations in infants under 12 months do not follow standard adult adiposity cutoffs.
  - All 911 infants in this dataset have `hypertension = 0` and `heart_disease = 0`.
- **Handling Strategy:** During schema validation, records with `age < 1.0` are handled via explicit boundary validation, and adult screening recommendations are enforced.

### 4.3 Sparse Demographic Subgroups (`gender = 'Other'`)
- **Audit Finding:** Only **18 records** out of 100,000 belong to the `'Other'` gender category (0.018% prevalence). All 18 are non-diabetic.
- **Handling Strategy:** The system issues an explicit sample size warning and documents that clinical generalization to this cohort is unsupported due to insufficient statistical power.

---

## 5. Preprocessing & Partitioning Protocol

To guarantee zero data leakage, the deduplicated dataset ($N = 96,146$) is partitioned using stratified sampling:

$$\text{Total Clean Records} = 96,146$$
- **Training Set (70%):** $N = 67,302$ (5,915 diabetic, 61,387 non-diabetic; positive prevalence $= 8.79\%$)
- **Validation Set (15%):** $N = 14,422$ (1,268 diabetic, 13,154 non-diabetic; positive prevalence $= 8.79\%$)
- **Test Set (Held-Out Generalization, 15%):** $N = 14,422$ (1,268 diabetic, 13,154 non-diabetic; positive prevalence $= 8.79\%$)

All scaling transformers (`StandardScaler`) and encoding matrices (`OneHotEncoder`) are fitted **strictly on the Training Set** and then applied to validation and test sets without recalculating parameters.

---

## 6. Engineered Features Dictionary

Five leakage-free clinical interaction features are generated during pipeline execution:

| Feature Name | Derivation Formula | Clinical Justification |
| :--- | :--- | :--- |
| `age_group` | Categorical discretization of `age` into `<18`, `18-35`, `36-50`, `51-65`, `65+` | Captures non-linear physiological risk inflections across life stages. |
| `bmi_category` | WHO adult BMI brackets: `Underweight`, `Normal`, `Overweight`, `Obese` | Standardized adiposity stratification for clinical triage. |
| `cardiometabolic_risk` | $\text{hypertension} + \text{heart\_disease} + (\text{bmi} \ge 30)$ | Composite vascular comorbidity score ranging from 0 to 3. |
| `glucose_hba1c_interaction` | $\text{blood\_glucose\_level} \times \text{HbA1c\_level}$ | Captures compound metabolic dysregulation (acute glycemic elevation paired with chronic elevation). |
| `age_bmi_interaction` | $\text{age} \times \text{bmi}$ | Synergistic metabolic burden reflecting duration and severity of elevated adiposity. |

---

## 7. Data Governance & Maintenance
- **Data Versioning:** Version-tracked raw and processed Parquet files with checksum verification.
- **Drift Monitoring:** Monitored weekly against incoming inference requests using Evidently AI (Wasserstein distance for continuous features; Jensen-Shannon divergence for categorical features).
- **Ethical Safeguards:** No personally identifiable information (PII) such as patient names, SSNs, or addresses is collected or stored.
