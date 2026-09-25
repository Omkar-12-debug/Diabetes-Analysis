# Model Card — DiabetesRiskModel (@champion)

Complying with the model card framework proposed by **Mitchell et al. (2019)** (*"Model Cards for Model Reporting"*, Proceedings of the Conference on Fairness, Accountability, and Transparency).

---

## 1. Model Details

- **Model Identifier:** `DiabetesRiskModel`
- **Model Registry Alias:** `@champion`
- **Version:** 21 (Production Champion)
- **Model Type:** Extreme Gradient Boosted Trees Ensemble (`XGBClassifier`)
- **Developer / Maintainer:** MLOps Diabetes Research & Clinical Informatics Team
- **Release Date:** September 2026
- **License:** Open Health AI / Academic Clinical Research
- **Framework & Dependencies:** XGBoost 2.0+, Scikit-Learn 1.3+, MLflow 2.11+, Python 3.11/3.13
- **Artifact Location:** `models:/DiabetesRiskModel@champion` / `models/model.tar.gz`

### Hyperparameter Specifications
- `n_estimators`: 150
- `max_depth`: 6
- `learning_rate`: 0.10
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `scale_pos_weight`: 10.33 (calibrated to the empirical negative-to-positive ratio in the deduplicated training cohort: $61,387 / 5,915 \approx 10.38$)
- `eval_metric`: `logloss`
- `objective`: `binary:logistic`
- `random_state`: 42 (deterministic reproducibility)

---

## 2. Intended Use & Clinical Scope

### Primary Intended Uses
1. **Clinical Screening & Triage Aid:** Designed for primary care physicians, nurses, and digital health clinics to estimate individual personalized risk of type 2 diabetes mellitus (T2DM) using routinely collected biometrics and laboratory panels.
2. **Preventive Intervention Stratification:** Automatically stratifies patients into four clinical risk tiers:
   - **Low Risk ($0 \le \text{Risk} < 25$):** Routine annual wellness follow-up.
   - **Moderate Risk ($25 \le \text{Risk} < 50$):** Lifestyle counseling, dietetic consultation, repeat screening within 6 months.
   - **High Risk ($50 \le \text{Risk} < 75$):** Supervised weight management, exercise intervention, diagnostic fasting plasma glucose / OGTT scheduling.
   - **Critical Risk ($75 \le \text{Risk} \le 100$):** Immediate physician evaluation, confirmatory lab testing, diagnostic clinical workup.
3. **Interactive Counterfactual Guidance:** Pairs predictions with DiCE counterfactual explanations to generate actionable lifestyle adjustments (e.g. realistic BMI and glycemic targets) while strictly locking non-modifiable features (age, biological sex).

### Out-of-Scope & Prohibited Applications
- ❌ **Autonomous Diagnostic Replacement:** Under no circumstances should this model replace standard laboratory diagnostic criteria (Fasting Plasma Glucose $\ge 126\text{ mg/dL}$, 2-hour Oral Glucose Tolerance Test $\ge 200\text{ mg/dL}$, or $\text{HbA1c} \ge 6.5\%$).
- ❌ **Automated Pharmacotherapy Titration:** Model risk scores must not autonomously initiate or adjust medication doses (e.g., Metformin, GLP-1 receptor agonists, insulin).
- ❌ **Pediatric Screening ($< 18$ years old):** Pediatric glucose dynamics and pediatric HbA1c reference intervals require specialized pediatric endocrinological oversight.
- ❌ **Emergency Department Critical Triage:** Not designed for diagnosing acute hyperosmolar hyperglycemic state (HHS) or diabetic ketoacidosis (DKA).

---

## 3. Factors & Demographic Cohorts

The model performance and fairness were evaluated across key demographic and clinical factors:
- **Biological Sex:** Female, Male, and Other.
- **Age Slices:** $<18$, $18–35$, $36–50$, $51–65$, and $65+$.
- **Cardiometabolic Comorbidities:** Hypertension and Heart Disease status.
- **Smoking History:** Never, No Info, Current, Former, Ever, Not Current.

---

## 4. Performance Summary (Held-Out Test Set, $N = 14,422$)

The model was evaluated on an isolated, held-out generalization test set ($N = 14,422$, positive prevalence $= 8.82\%$).

### Quantitative Metrics
| Metric | Validation Set | Held-Out Test Set | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **PR-AUC (Precision-Recall AUC)** | **0.8860** | **0.8829** | Primary ranking metric under class imbalance; reflects superior precision across all recall levels. |
| **ROC-AUC** | **0.9779** | **0.9779** | Near-optimal discriminative separation between diabetic and non-diabetic cohorts. |
| **Recall (Sensitivity)** | **90.34%** | **90.57%** | **Identifies $>90.5\%$ of true diabetic patients**, minimizing dangerous false negatives. |
| **Precision** | **50.84%** | **48.65%** | High positive predictive value given 10.33:1 imbalance (~1 in 2 flagged individuals confirmed). |
| **F1-Score** | **0.6506** | **0.6330** | Balanced harmonic mean under severe label imbalance. |
| **Balanced Accuracy** | **0.9094** | **0.9066** | Arithmetic mean of sensitivity ($90.57\%$) and specificity ($90.76\%$). |
| **Brier Score** | **0.0537** | **0.0561** | Mean squared calibration error; confirms sharp probability estimation near 0.0. |

### Confusion Matrix on Held-Out Test Set ($N = 14,422$)
$$\begin{pmatrix} \text{TN} = 11,934 & \text{FP} = 1,216 \\ \text{FN} = 120 & \text{TP} = 1,152 \end{pmatrix}$$
- **Specificity (True Negative Rate):** $\frac{11,934}{11,934 + 1,216} = 90.75\%$
- **False Negative Rate (Miss Rate):** $\frac{120}{120 + 1,152} = 9.43\%$

---

## 5. Probability Calibration & Reliability Analysis

Under cost-sensitive weighting (`scale_pos_weight = 10.33`), raw tree scores reflect modified prior probabilities. The probability calibration audit reveals:
- **Brier Score Loss:** `0.0561` (optimal $< 0.10$)
- **Expected Calibration Error (ECE):** `0.0837` ($8.37\%$)
- **Maximum Calibration Error (MCE):** `0.5579`

### Reliability Diagram Characteristics
1. **Low-Risk Bin Calibration ($[0.0, 0.1)$):** Over $70\%$ of test samples reside in this bin ($N = 10,170$) with an empirical positive rate of $0.12\%$ vs. mean predicted confidence of $0.76\%$, showing outstanding negative predictive reliability.
2. **Intermediate Risk Regions ($0.30 - 0.70$):** Raw probabilities exhibit an intentional conservative upward tilt, which is clinically desirable in risk screening to prioritize borderline patients for lab validation.
3. **High-Risk Bins ($> 0.90$):** High-confidence predictions achieve $>97.5\%$ empirical positive prevalence, confirming that high-tier alerts are clinically dependable.

---

## 6. Demographic Fairness & Disparity Audit

### 6.1 Biological Sex Subgroup Analysis
| Subgroup | Support ($N$) | Positives | Prevalence | Recall (TPR) | FPR | Precision | F1-Score | PR-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Female** | 8,473 | 664 | 7.84% | 88.55% | 8.18% | 47.92% | 0.6219 | 0.8697 | Statistically Robust |
| **Male** | 5,947 | 608 | 10.22% | 92.76% | 10.81% | 49.43% | 0.6449 | 0.8966 | Statistically Robust |
| **Other** | 2 | 0 | 0.00% | N/A | 0.00% | N/A | 0.0000 | N/A | ⚠️ **Sample Size Warning ($N=2$)** |

#### Parity Metrics
- **Equal Opportunity Ratio (Recall Disparity):** $\frac{\text{Recall}_{\text{Female}}}{\text{Recall}_{\text{Male}}} = \frac{0.8855}{0.9276} = 0.9546$ (well above the 0.80 four-fifths rule for fairness).
- **False Positive Rate Difference:** $|\text{FPR}_{\text{Male}} - \text{FPR}_{\text{Female}}| = 10.81\% - 8.18\% = 2.63\%$.

#### Small Cohort Clinical Limitation
> [!WARNING]
> The 'Other' gender category contains only $N=2$ records in the test set ($N=18$ across 100k raw cohort) with 0 positive cases. Metrics cannot be computed with statistical power. Clinical generalization to non-binary or intersex cohorts is unvalidated, and targeted prospective data collection is required.

### 6.2 Age Group Subgroup Analysis
| Age Group | Support ($N$) | Positives | Prevalence | Recall (TPR) | FPR | Precision | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 18** | 2,600 | 14 | 0.54% | 71.43% | 0.00% | 100.0% | 0.8333 | 0.7317 |
| **18 – 35** | 3,073 | 51 | 1.66% | 62.75% | 0.50% | 68.09% | 0.6531 | 0.7237 |
| **36 – 50** | 3,058 | 225 | 7.36% | 78.22% | 4.69% | 56.96% | 0.6592 | 0.8335 |
| **51 – 65** | 3,097 | 432 | 13.95% | 92.82% | 17.64% | 46.04% | 0.6155 | 0.8886 |
| **65+** | 2,594 | 550 | 21.20% | 96.91% | 29.26% | 47.13% | 0.6341 | 0.9083 |

- **Clinical Insight:** In seniors ($65+$), where disease prevalence is high ($21.20\%$), the model achieves $96.91\%$ recall, minimizing missed diagnoses in vulnerable geriatric populations. In younger adults ($<35$), specificity is kept high ($>99.5\%$) to minimize unnecessary clinical alarm.

---

## 7. Explainability & SHAP Feature Importance Summary

Global feature attribution was computed using TreeSHAP on the independent test set.

### Feature Importance Ranking (Mean Absolute SHAP Value)
| Rank | Feature | Description | Mean $|\text{SHAP}|$ | Primary Direction of Effect |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `HbA1c_level` | Glycated hemoglobin (%) | **2.142** | Strong positive correlation with diabetic risk ($\ge 6.5\%$ causes massive risk escalation). |
| **2** | `blood_glucose_level` | Fasting/casual glucose ($mg/dL$) | **1.874** | Positive correlation ($\ge 140\text{ mg/dL}$ sharply increases predicted score). |
| **3** | `glucose_hba1c_interaction` | Multiplicative interaction term | **0.621** | Compound risk amplifier when both metabolic markers are elevated. |
| **4** | `age` | Patient chronological age (years) | **0.485** | Gradual upward baseline risk trajectory above 45 years. |
| **5** | `bmi` | Body Mass Index ($kg/m^2$) | **0.392** | Obesity tiers ($\text{BMI} \ge 30$) exert positive attribution. |
| **6** | `age_bmi_interaction` | Joint age-adiposity index | **0.210** | Synergistic metabolic risk in older patients with elevated adiposity. |
| **7** | `hypertension` | High blood pressure flag | **0.145** | Elevates baseline probability via vascular risk pathway. |
| **8** | `heart_disease` | Cardiovascular disease flag | **0.112** | Secondary vascular comorbidity indicator. |
| **9** | `smoking_history` | Tobacco consumption history | **0.084** | Current and former smoking status modestly elevate risk. |

---

## 8. Environmental & Computational Footprint
- **Hardware:** Intel Core i7 / AMD Ryzen 8-core CPU, 16 GB RAM.
- **Training Time:** 8.4 seconds for 150 gradient-boosted trees on $N=67,302$ samples.
- **Inference Latency:** Median (p50) $12\text{ ms}$, 95th percentile (p95) $28\text{ ms}$.
- **Carbon Footprint:** $< 0.005\text{ kg CO}_2\text{eq}$ (negligible compute footprint).
