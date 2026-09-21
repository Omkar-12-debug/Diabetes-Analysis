/**
 * Maps raw model feature names (from SHAP/preprocessing) into plain-English
 * patient-friendly labels that non-technical users can understand.
 */

const FEATURE_MAP: Record<string, string> = {
  // Original raw features
  blood_glucose_level: "Blood Sugar Level",
  HbA1c_level: "HbA1c (Average Blood Sugar)",
  bmi: "Body Mass Index (BMI)",
  age: "Age",
  gender: "Biological Sex",
  hypertension: "High Blood Pressure",
  heart_disease: "Heart Disease History",
  smoking_history: "Smoking Status",

  // Engineered features
  age_group: "Age Bracket",
  bmi_category: "Weight Category",
  cardiometabolic_risk: "Heart & Metabolic Health Score",
  glucose_hba1c_interaction: "Blood Sugar & HbA1c Combined Effect",
  age_bmi_interaction: "Age & Weight Combined Effect",

  // Preprocessed numeric features (num__ prefix)
  "num__blood_glucose_level": "Blood Sugar Level",
  "num__HbA1c_level": "HbA1c (Average Blood Sugar)",
  "num__bmi": "Body Mass Index (BMI)",
  "num__age": "Age",
  "num__cardiometabolic_risk": "Heart & Metabolic Health Score",
  "num__glucose_hba1c_interaction": "Blood Sugar & HbA1c Combined Effect",
  "num__age_bmi_interaction": "Age & Weight Combined Effect",

  // Preprocessed categorical features (cat__ prefix)
  "cat__gender_Female": "Female",
  "cat__gender_Male": "Male",
  "cat__gender_Other": "Other Gender",
  "cat__smoking_history_No Info": "Smoking Status Unknown",
  "cat__smoking_history_current": "Current Smoker",
  "cat__smoking_history_ever": "Has Smoked Before",
  "cat__smoking_history_former": "Former Smoker",
  "cat__smoking_history_never": "Non-Smoker",
  "cat__smoking_history_not current": "Not Currently Smoking",
  "cat__hypertension_0": "No High Blood Pressure",
  "cat__hypertension_1": "Has High Blood Pressure",
  "cat__heart_disease_0": "No Heart Disease",
  "cat__heart_disease_1": "Has Heart Disease",
  "cat__age_group_Middle": "Middle-Aged (30–49)",
  "cat__age_group_Senior": "Senior (50–64)",
  "cat__age_group_Elderly": "Elderly (65+)",
  "cat__age_group_Young": "Young Adult (18–29)",
  "cat__bmi_category_Overweight": "Overweight",
  "cat__bmi_category_Obese": "Obese",
  "cat__bmi_category_Normal": "Normal Weight",
  "cat__bmi_category_Underweight": "Underweight",
};

/**
 * Convert a raw model feature string into a patient-friendly human label.
 * Falls back to a cleaned-up version of the raw name if not in our map.
 */
export function formatFeatureName(rawName: string): string {
  if (FEATURE_MAP[rawName]) return FEATURE_MAP[rawName];

  // Auto-cleanup: remove num__, cat__, replace _ with space, title-case
  let cleaned = rawName
    .replace(/^num__/, "")
    .replace(/^cat__/, "")
    .replace(/_/g, " ");

  // Title case each word
  cleaned = cleaned
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

  return cleaned;
}

/**
 * Format a raw feature value into a patient-readable string.
 */
export function formatFeatureValue(
  rawName: string,
  value: string | number | null | undefined
): string {
  if (value === null || value === undefined) return "—";

  const numVal = typeof value === "number" ? value : parseFloat(String(value));

  if (rawName.includes("blood_glucose")) return `${numVal} mg/dL`;
  if (rawName.includes("HbA1c")) return `${numVal.toFixed(1)}%`;
  if (rawName.includes("bmi") && !rawName.includes("category"))
    return `${numVal.toFixed(1)} kg/m²`;
  if (rawName === "age") return `${numVal} years`;

  return String(value);
}
