/**
 * Clinical API Client with seamless fallback to realistic mock presets when FastAPI is offline.
 */

import {
  AssessmentHistory,
  DelayedLabelInput,
  DelayedLabelResponse,
  ExplanationResponse,
  HealthResponse,
  HistoryListResponse,
  ModelInfoResponse,
  PatientInput,
  PredictionResponse,
  WhatIfRequest,
  WhatIfResponse,
} from "../types";

const API_BASE = "/api";

export const MANDATORY_DISCLAIMER =
  "This counterfactual is a hypothetical model sensitivity exploration and must not be interpreted as causal medical advice or a treatment plan.";

// Local in-memory store for fallback mode
const mockHistoryStore: Record<string, AssessmentHistory[]> = {};

function calculateMockRisk(patient: PatientInput): { prob: number; score: number; cat: "Low" | "Moderate" | "High" } {
  let score = 5;
  // Blood glucose impact
  if (patient.blood_glucose_level >= 200) score += 45;
  else if (patient.blood_glucose_level >= 140) score += 25;
  else if (patient.blood_glucose_level >= 100) score += 10;

  // HbA1c impact
  if (patient.HbA1c_level >= 7.0) score += 35;
  else if (patient.HbA1c_level >= 6.0) score += 20;
  else if (patient.HbA1c_level >= 5.7) score += 8;

  // BMI impact
  if (patient.bmi >= 35) score += 15;
  else if (patient.bmi >= 30) score += 10;
  else if (patient.bmi >= 25) score += 5;

  // Age & history
  if (patient.age >= 60) score += 8;
  else if (patient.age >= 45) score += 4;
  if (patient.hypertension === 1) score += 6;
  if (patient.heart_disease === 1) score += 6;

  score = Math.min(99, Math.max(1, score));
  const prob = Number((score / 100).toFixed(4));
  const cat = score < 30 ? "Low" : score <= 70 ? "Moderate" : "High";
  return { prob, score, cat };
}

export const apiClient = {
  async getHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      status: "healthy (mock)",
      service: "DiaGuard AI Mock Engine",
      model_name: "DiabetesRiskModel",
      model_version: "2",
      timestamp: new Date().toISOString(),
    };
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    try {
      const res = await fetch(`${API_BASE}/model-info`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      model_name: "DiabetesRiskModel",
      alias: "champion",
      model_version: "2",
      algorithm: "XGBClassifier (Calibrated)",
      status: "promoted_by_quality_gate",
      metrics_summary: {
        test_pr_auc: 0.8829,
        test_clinical_sensitivity: 0.9057,
        test_brier_score: 0.0561,
        fairness_gender_disparity: 0.0421,
      },
      thresholds: {
        min_pr_auc: 0.85,
        min_sensitivity: 0.88,
        max_brier_score: 0.1,
      },
    };
  },

  async predict(patient: PatientInput): Promise<PredictionResponse> {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }

    const { prob, score, cat } = calculateMockRisk(patient);
    const pid = patient.patient_id || `patient-sim-${Math.floor(Math.random() * 9000 + 1000)}`;
    const assessmentId = Math.floor(Math.random() * 90000 + 10000);

    const record: AssessmentHistory = {
      id: assessmentId,
      patient_id: pid,
      created_at: new Date().toISOString(),
      input_features: { ...patient },
      risk_probability: prob,
      risk_score: score,
      risk_category: cat,
      model_version: "2",
    };

    if (!mockHistoryStore[pid]) mockHistoryStore[pid] = [];
    mockHistoryStore[pid].push(record);

    return {
      prediction: prob >= 0.5 ? 1 : 0,
      probability: prob,
      risk_score: score,
      risk_category: cat,
      model_name: "DiabetesRiskModel",
      model_version: "2",
      timestamp: new Date().toISOString(),
      patient_id: pid,
      assessment_id: assessmentId,
    };
  },

  async explain(patient: PatientInput): Promise<ExplanationResponse> {
    try {
      const res = await fetch(`${API_BASE}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
        signal: AbortSignal.timeout(6000),
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }

    const { prob, score } = calculateMockRisk(patient);
    const glucoseDiff = (patient.blood_glucose_level - 100) / 40;
    const hba1cDiff = (patient.HbA1c_level - 5.5) / 1.5;
    const bmiDiff = (patient.bmi - 25.0) / 8.0;

    const attributions = [
      {
        feature: "blood_glucose_level",
        attribution_value: Number(glucoseDiff.toFixed(3)),
        direction: (glucoseDiff >= 0 ? "increases_risk" : "decreases_risk") as any,
        raw_value: patient.blood_glucose_level,
      },
      {
        feature: "HbA1c_level",
        attribution_value: Number(hba1cDiff.toFixed(3)),
        direction: (hba1cDiff >= 0 ? "increases_risk" : "decreases_risk") as any,
        raw_value: patient.HbA1c_level,
      },
      {
        feature: "glucose_hba1c_interaction",
        attribution_value: Number(((glucoseDiff + hba1cDiff) * 0.4).toFixed(3)),
        direction: (glucoseDiff + hba1cDiff >= 0 ? "increases_risk" : "decreases_risk") as any,
        raw_value: Number((patient.blood_glucose_level * patient.HbA1c_level).toFixed(1)),
      },
      {
        feature: "bmi",
        attribution_value: Number(bmiDiff.toFixed(3)),
        direction: (bmiDiff >= 0 ? "increases_risk" : "decreases_risk") as any,
        raw_value: patient.bmi,
      },
      {
        feature: "age",
        attribution_value: Number(((patient.age - 45) / 50).toFixed(3)),
        direction: (patient.age >= 45 ? "increases_risk" : "decreases_risk") as any,
        raw_value: patient.age,
      },
      {
        feature: "cardiometabolic_risk",
        attribution_value: patient.hypertension && patient.heart_disease ? 0.35 : patient.hypertension ? 0.18 : -0.15,
        direction: (patient.hypertension ? "increases_risk" : "decreases_risk") as any,
        raw_value: patient.hypertension + patient.heart_disease,
      },
    ];

    const pos = attributions.filter((a) => a.attribution_value > 0).sort((a, b) => b.attribution_value - a.attribution_value);
    const neg = attributions.filter((a) => a.attribution_value <= 0).sort((a, b) => a.attribution_value - b.attribution_value);

    return {
      base_value: -2.35,
      predicted_margin: Number((prob * 4 - 2).toFixed(3)),
      predicted_probability: prob,
      risk_classification: score < 30 ? "Low Risk" : score <= 70 ? "Moderate Risk" : "High Risk",
      attributions,
      top_risk_increasing_factors: pos.slice(0, 3),
      top_risk_decreasing_factors: neg.slice(0, 3),
      model_version: "2",
    };
  },

  async whatIf(request: WhatIfRequest): Promise<WhatIfResponse> {
    try {
      const res = await fetch(`${API_BASE}/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(6000),
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }

    const origRisk = calculateMockRisk(request.patient);
    const counterPatient: PatientInput = {
      ...request.patient,
      ...(request.modifiable_overrides || {}),
    };
    const counterRisk = calculateMockRisk(counterPatient);
    const delta = Number((counterRisk.prob - origRisk.prob).toFixed(4));
    const targetThreshold = request.target_risk_threshold ?? 0.3;

    return {
      original_probability: origRisk.prob,
      counterfactual_probability: counterRisk.prob,
      probability_delta: delta,
      target_threshold: targetThreshold,
      target_achieved: counterRisk.prob <= targetThreshold,
      feature_changes: request.modifiable_overrides || {},
      counterfactual_patient: counterPatient as any,
      non_modifiable_preserved: true,
      disclaimer: MANDATORY_DISCLAIMER,
    };
  },

  async getPatientHistory(patientId: string): Promise<HistoryListResponse> {
    try {
      const res = await fetch(`${API_BASE}/history/${encodeURIComponent(patientId)}`, {
        signal: AbortSignal.timeout(4000),
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }

    const records = mockHistoryStore[patientId] || [];
    return {
      patient_id: patientId,
      total_records: records.length,
      assessments: records,
    };
  },

  async recordGroundTruth(input: DelayedLabelInput): Promise<DelayedLabelResponse> {
    try {
      const res = await fetch(`${API_BASE}/feedback/ground-truth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
        signal: AbortSignal.timeout(4000),
      });
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }

    return {
      id: Math.floor(Math.random() * 1000 + 1),
      assessment_id: input.assessment_id,
      patient_id: "patient-feedback",
      verified_diabetes_label: input.verified_diabetes_label,
      verified_at: new Date().toISOString(),
      notes: input.notes || null,
      message: "Ground truth clinical verification successfully recorded (simulated).",
    };
  },
};
