import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { AssessmentForm, PRESETS } from "./components/AssessmentForm";
import { RiskGauge } from "./components/RiskGauge";
import { ShapWaterfall } from "./components/ShapWaterfall";
import { WhatIfSimulator } from "./components/WhatIfSimulator";
import { PatientHistory } from "./components/PatientHistory";
import {
  AssessmentHistory,
  ExplanationResponse,
  HealthResponse,
  ModelInfoResponse,
  PatientInput,
  PredictionResponse,
} from "./types";
import { apiClient } from "./api/client";
import { Shield } from "lucide-react";

export const App: React.FC = () => {
  // Active clinical patient inputs
  const [patient, setPatient] = useState<PatientInput>(PRESETS.borderline);

  // Inference, explainability, and history states
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [history, setHistory] = useState<AssessmentHistory[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  // System status
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  // Refresh history for active patient
  const refreshHistory = async (patientId: string) => {
    try {
      const histResp = await apiClient.getPatientHistory(patientId);
      setHistory(histResp.assessments);
    } catch {
      // Ignored
    }
  };

  // Run full assessment (predict + explain + history)
  const runAssessment = async (currentPatient: PatientInput = patient) => {
    setLoading(true);
    try {
      // Execute inference and explanation concurrently
      const [predRes, expRes] = await Promise.all([
        apiClient.predict(currentPatient),
        apiClient.explain(currentPatient),
      ]);
      setPrediction(predRes);
      setExplanation(expRes);

      if (predRes.patient_id) {
        await refreshHistory(predRes.patient_id);
      }
    } catch (err) {
      console.error("Assessment error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Initial startup: check health, model info, and run initial demonstration
  useEffect(() => {
    const initApp = async () => {
      try {
        const [h, m] = await Promise.all([apiClient.getHealth(), apiClient.getModelInfo()]);
        setHealth(h);
        setModelInfo(m);
        setIsLive(!h.status.includes("mock"));
      } catch {
        setIsLive(false);
      }
      // Run initial baseline assessment
      runAssessment(PRESETS.borderline);
    };
    initApp();
  }, []);

  return (
    <div className="min-h-screen bg-clinical-dark text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Header health={health} modelInfo={modelInfo} isLive={isLive} />

      {/* Main Clinical Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Top Section: Form + Risk Gauge */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Form (5 cols) */}
          <div className="lg:col-span-5">
            <AssessmentForm
              patient={patient}
              onChange={setPatient}
              onSubmit={() => runAssessment(patient)}
              loading={loading}
            />
          </div>

          {/* Right Column: Risk Gauge & SHAP (7 cols) */}
          <div className="lg:col-span-7 space-y-8">
            <RiskGauge prediction={prediction} loading={loading} />
            <ShapWaterfall explanation={explanation} loading={loading} />
          </div>
        </div>

        {/* Middle Section: Counterfactual What-If Simulator */}
        <div className="w-full">
          <WhatIfSimulator basePatient={patient} />
        </div>

        {/* Bottom Section: Longitudinal Patient Risk Trajectory */}
        <div className="w-full">
          <PatientHistory
            patientId={patient.patient_id || "PT-BORDER-02"}
            history={history}
            onRefresh={() => refreshHistory(patient.patient_id || "PT-BORDER-02")}
          />
        </div>
      </main>

      {/* Medical & MLOps Footer */}
      <footer className="glass-panel border-t border-clinical-border py-6 px-4 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-slate-400">
            <Shield className="w-4 h-4 text-clinical-cyan" />
            <span>DiaGuard AI CDSS Platform • Calibrated Clinical Decision Support</span>
          </div>
          <div className="flex items-center gap-4 text-slate-500">
            <span>Model Registry: <code className="text-slate-400">@champion (XGBoost)</code></span>
            <span>PR-AUC: <code className="text-emerald-400 font-bold">0.8829</code></span>
            <span>Recall: <code className="text-emerald-400 font-bold">0.9057</code></span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
