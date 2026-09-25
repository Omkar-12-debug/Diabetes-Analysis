# Data Card: Diabetes Prediction Dataset

## 1. Dataset Overview & Provenance

- **Dataset Name**: Diabetes Prediction Dataset
- **Repository Path**: `data/raw/diabetes_prediction_dataset.csv`
- **Volume**: 100,000 records
- **Dimensionality**: 9 columns (8 predictive features + 1 binary target label)
- **Domain**: Healthcare / Chronic Disease Risk Prediction
- **Provenance**: Derived from electronic health records and demographic health survey aggregations. The dataset provides structured clinical and demographic markers commonly assessed in outpatient risk stratification.
- **License**: Public domain / Open Research (CC BY 4.0 / Public Dataset)
- **Primary Objective**: Early identification and risk scoring of diabetes mellitus in diverse adult and adolescent populations.

---

## 2. Feature Definitions & Schema

| Feature Name | Data Type | Physical / Clinical Meaning | Measurement Unit | Permitted / Expected Range | Null Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gender` | Categorical (str) | Biological sex / gender identity of patient | N/A | `['Female', 'Male', 'Other']` | 0 |
| `age` | Numerical (float) | Age of the individual at examination | Years | [0.0, 120.0] | 0 |
| `hypertension` | Binary (int) | Clinical history of persistent high blood pressure | Flag | {0, 1} (0: No, 1: Yes) | 0 |
| `heart_disease` | Binary (int) | Medical history of cardiovascular disease / CAD | Flag | {0, 1} (0: No, 1: Yes) | 0 |
| `smoking_history` | Categorical (str) | Self-reported or documented tobacco exposure | N/A | `['never', 'No Info', 'current', 'former', 'ever', 'not current']` | 0 |
| `bmi` | Numerical (float) | Body Mass Index ($weight [kg] / height [m]^2$) | $kg/m^2$ | [10.0, 100.0] | 0 |
| `HbA1c_level` | Numerical (float) | Glycated hemoglobin test measuring 3-month average blood glucose | % of total hemoglobin | [3.0, 20.0] | 0 |
| `blood_glucose_level` | Numerical (int/float) | Instantaneous blood glucose concentration | $mg/dL$ | [30.0, 500.0] | 0 |
| `diabetes` | Binary (int) | Confirmed diagnosis of diabetes mellitus (Target) | Flag | {0, 1} (0: Non-diabetic, 1: Diabetic) | 0 |

---

## 3. Class Distribution & Imbalance Analysis

| Target Class | Description | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | Non-diabetic | 91,500 | 91.50% |
| **1** | Diabetic | 8,500 | 8.50% |
| **Total** | Full Cohort | 100,000 | 100.00% |

- **Imbalance Ratio**: ~10.76 : 1 (Non-diabetic to Diabetic).
- **ML Implications**:
  - Accuracy is an inappropriate evaluation metric due to high baseline prevalence of the negative class. A naive dummy classifier achieves 91.5% accuracy by always predicting class 0.
  - Evaluation must prioritize **PR-AUC (Precision-Recall Area Under Curve)**, **Recall / Sensitivity** (to minimize false negatives in patient risk detection), **F1-Score / F2-Score**, and **ROC-AUC**.
  - Pipeline modeling must leverage techniques such as class-weighted loss functions (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in logistic regression/random forest), calibrated threshold tuning, or resampled validation partitions.

---

## 4. Data Quality Audit & Identified Anomalies

An automated quality audit of the raw dataset identified the following specific findings:

### 4.1 Exact Duplicate Records
- **Count**: 3,854 rows (3.854% of total dataset).
- **Assessment**: Multiple identical rows occur across all 9 features including target label. In clinical datasets, this commonly stems from administrative re-entries or identical multi-visit record captures.
- **Phase 3 Recommendation**: Deduplicate records prior to train-test splitting to prevent severe data leakage between training and evaluation folds.

### 4.2 Sub-Year / Infant Records (`age < 1.0`)
- **Count**: 911 records (0.911% of dataset).
- **Range**: Minimum age observed is 0.08 years (~1 month).
- **Assessment**: Glycated hemoglobin (HbA1c) and adult BMI scales are clinically calibrated for adult/pediatric age groups rather than neonates. None of these 911 infants possess hypertension or heart disease.
- **Phase 3 Recommendation**: Evaluate stratified handling or dedicated cohort filtering during preprocessing depending on clinical deployment scope.

### 4.3 Conflicting Feature Combinations with Discordant Labels
- **Count**: 91 distinct feature combinations (affecting 223 total rows).
- **Description**: Distinct patient instances possess identical demographic, biomarker, and lifestyle values across all 8 input features, yet are labeled with opposing `diabetes` outcomes (one record marked as `0` and another marked as `1`).
- **Clinical Rationale**: Diabetes etiology involves unmeasured genetic, medication, or dietary factors not captured in the 8 features.
- **Phase 3 Recommendation**: Document label noise; resolve or filter ambiguous label conflicts during data preparation to maintain clean ground-truth validation sets.

### 4.4 Demographic Subgroup Sparsity & Missingness
- **'Other' Gender**:
  - Only **18 records** (0.018% of dataset).
  - Extreme underrepresentation impairs statistical generalizability for this subgroup.
- **'No Info' Smoking History**:
  - **35,816 records** (35.816% of dataset).
  - Rather than random missingness, 'No Info' represents unrecorded documentation or absence of clinical inquiry.
  - Must be preserved as an explicit informative category rather than dropped.

---

## 5. Intended Use & Deployment Scope

### 5.1 Approved Intended Uses
- **Population Risk Stratification**: Assisting clinics and healthcare networks in prioritizing patients for secondary laboratory follow-up (fasting glucose or oral glucose tolerance test).
- **Clinical Decision Support (CDS) Assistance**: Providing probabilistic risk indicators alongside clinician reviews during routine primary care consultations.
- **Lifestyle Intervention Triage**: Identifying high-risk pre-diabetic individuals who would benefit from dietary, lifestyle, or weight management interventions.

### 5.2 Out-of-Scope Uses & Prohibited Applications
- **Standalone Diagnostic Tool**: Under no circumstances should this model replace clinical diagnostic criteria (such as formal laboratory venous blood tests, oral glucose tolerance tests, or licensed medical practitioner judgment).
- **Automated Care Denial or Insurance Discrimination**: Prohibited from being utilized for automated health insurance policy denial, underwriting penalties, or employment screening.
- **Pediatric Neonatal Diagnosis**: The dataset is unsuited for neonatal or infantile metabolic disorder diagnosis.

---

## 6. Responsible AI & Demographic Considerations

1. **Subgroup Performance Disparity**:
   - The 'Other' gender cohort ($N=18$) must not be used to infer reliable performance metrics for non-binary or gender-diverse individuals without prospective clinical validation.
2. **Missing Information as a Bias Source**:
   - High rates of 'No Info' smoking history (35.8%) may correlate with lower interaction with healthcare systems or socioeconomic factors.
3. **Threshold Calibration & False Negatives**:
   - In healthcare screening, a False Negative (failing to detect an individual with diabetes) carries severe clinical risk of unmanaged microvascular complications (neuropathy, retinopathy, nephropathy).
   - Downstream serving architectures must calibrate classification thresholds to maintain high clinical sensitivity (recall $\ge 0.85$).
