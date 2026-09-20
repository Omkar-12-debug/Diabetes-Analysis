/**
 * TypeScript definitions strictly aligned with FastAPI Pydantic v2 schemas.
 */

export type Gender = "Female" | "Male" | "Other";
export type BinaryFlag = 0 | 1;
export type SmokingHistory = "never" | "No Info" | "current" | "former" | "ever" | "not current";
export type RiskCategory = "Low" | "Moderate" | "High";

export interface PatientInput {
  patient_id?: string;
  gender: Gender;
  age: number;
  hypertension: BinaryFlag;
  heart_disease: BinaryFlag;
  smoking_history: SmokingHistory;
  bmi: number;
  HbA1c_level: number;
  blood_glucose_level: number;
}

export interface PredictionResponse {
  prediction: number;
  probability: number;
  risk_score: number;
  risk_category: RiskCategory;
  model_name: string;
  model_version: string;
  timestamp: string;
  patient_id?: string;
  assessment_id?: number;
}

export interface FeatureAttribution {
  feature: string;
  attribution_value: number;
  direction: "increases_risk" | "decreases_risk";
  raw_value?: string | number | null;
}

export interface ExplanationResponse {
  base_value: number;
  predicted_margin: number;
  predicted_probability: number;
  risk_classification: string;
  attributions: FeatureAttribution[];
  top_risk_increasing_factors: FeatureAttribution[];
  top_risk_decreasing_factors: FeatureAttribution[];
  model_version: string;
}

export interface WhatIfRequest {
  patient: PatientInput;
  modifiable_overrides?: Partial<Pick<PatientInput, "bmi" | "HbA1c_level" | "blood_glucose_level" | "smoking_history">>;
  target_risk_threshold?: number;
}

export interface WhatIfResponse {
  original_probability: number;
  counterfactual_probability: number;
  probability_delta: number;
  target_threshold: number;
  target_achieved: boolean;
  feature_changes: Record<string, string | number>;
  counterfactual_patient: Record<string, string | number>;
  non_modifiable_preserved: boolean;
  disclaimer: string;
}

export interface AssessmentHistory {
  id: number;
  patient_id: string;
  created_at: string;
  input_features: Record<string, any>;
  risk_probability: number;
  risk_score: number;
  risk_category: RiskCategory;
  model_version: string;
  verified_diabetes_label?: number | null;
  verified_at?: string | null;
  notes?: string | null;
}

export interface HistoryListResponse {
  patient_id?: string | null;
  total_records: number;
  assessments: AssessmentHistory[];
}

export interface DelayedLabelInput {
  assessment_id: number;
  verified_diabetes_label: 0 | 1;
  notes?: string;
}

export interface DelayedLabelResponse {
  id: number;
  assessment_id: number;
  patient_id: string;
  verified_diabetes_label: number;
  verified_at: string;
  notes?: string | null;
  message: string;
}

export interface ModelInfoResponse {
  model_name: string;
  alias: string;
  model_version: string;
  algorithm: string;
  status: string;
  metrics_summary: Record<string, any>;
  thresholds: Record<string, any>;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_name: string;
  model_version: string;
  timestamp: string;
}
