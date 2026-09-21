import React, { useState, useCallback } from "react";
import {
  SlidersHorizontal,
  RefreshCw,
  ArrowDown,
  ArrowUp,
  Minus,
  AlertTriangle,
  TrendingDown,
} from "lucide-react";
import { PatientInput, PredictionResponse, WhatIfResponse } from "../types";
import { apiClient } from "../api/client";
import { formatFeatureName } from "../utils/featureFormatter";

interface SimulatorViewProps {
  patient: PatientInput;
  prediction: PredictionResponse;
}

export const SimulatorView: React.FC<SimulatorViewProps> = ({ patient, prediction }) => {
  const [bmi, setBmi] = useState(patient.bmi);
  const [hba1c, setHba1c] = useState(patient.HbA1c_level);
  const [glucose, setGlucose] = useState(patient.blood_glucose_level);
  const [smokingHistory, setSmokingHistory] = useState<string>(patient.smoking_history);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WhatIfResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.whatIf({
        patient,
        modifiable_overrides: {
          bmi,
          HbA1c_level: hba1c,
          blood_glucose_level: glucose,
          smoking_history: smokingHistory as PatientInput["smoking_history"],
        },
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Simulation failed.");
    } finally {
      setLoading(false);
    }
  }, [patient, bmi, hba1c, glucose, smokingHistory]);

  const resetSliders = () => {
    setBmi(patient.bmi);
    setHba1c(patient.HbA1c_level);
    setGlucose(patient.blood_glucose_level);
    setSmokingHistory(patient.smoking_history);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <SlidersHorizontal className="w-6 h-6 text-primary-600" />
          What-If Health Simulator
        </h2>
        <p className="text-sm text-slate-500">
          Adjust the sliders below to explore how lifestyle changes could affect your diabetes risk.
          Only modifiable health factors are shown — things you can actually change.
        </p>
      </div>

      {/* Current baseline reminder */}
      <div className="card-padded bg-slate-50 flex items-center gap-4">
        <div className="text-sm text-slate-600">
          Your current risk score: <strong className="text-lg text-slate-800">{prediction.risk_score}%</strong>{" "}
          <span className={`font-semibold ${
            prediction.risk_category === "Low" ? "text-risk-low" :
            prediction.risk_category === "Moderate" ? "text-risk-mod" :
            "text-risk-high"
          }`}>
            ({prediction.risk_category})
          </span>
        </div>
      </div>

      {/* Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <SliderCard
          label="Body Mass Index (BMI)"
          help="Target a healthier weight range"
          value={bmi}
          originalValue={patient.bmi}
          min={12}
          max={65}
          step={0.1}
          unit="kg/m²"
          onChange={setBmi}
        />
        <SliderCard
          label="HbA1c Level"
          help="Lower with diet, exercise, and medication"
          value={hba1c}
          originalValue={patient.HbA1c_level}
          min={3.5}
          max={15}
          step={0.1}
          unit="%"
          onChange={setHba1c}
        />
        <SliderCard
          label="Blood Glucose Level"
          help="Manage with diet and medication"
          value={glucose}
          originalValue={patient.blood_glucose_level}
          min={50}
          max={350}
          step={1}
          unit="mg/dL"
          onChange={setGlucose}
        />
        <div className="card-padded">
          <label className="block text-sm font-semibold text-slate-700 mb-1">Smoking Status</label>
          <p className="text-xs text-slate-400 mb-3">
            Quitting smoking can improve metabolic health
          </p>
          <select
            value={smokingHistory}
            onChange={(e) => setSmokingHistory(e.target.value)}
            className="input-field"
          >
            <option value="never">Never smoked</option>
            <option value="former">Former smoker</option>
            <option value="current">Current smoker</option>
            <option value="not current">Not currently smoking</option>
            <option value="ever">Has smoked before</option>
            <option value="No Info">Prefer not to say</option>
          </select>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          onClick={runSimulation}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? (
            <><RefreshCw className="w-4 h-4 animate-spin" /> Simulating...</>
          ) : (
            <><TrendingDown className="w-4 h-4" /> Run Simulation</>
          )}
        </button>
        <button onClick={resetSliders} className="btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Reset to Original
        </button>
      </div>

      {error && (
        <div className="card-padded border-l-4 border-l-red-400 bg-red-50/60 text-sm text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="card-padded space-y-5">
          <h3 className="font-semibold text-slate-900 text-lg">Simulation Results</h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatBox
              label="Original Risk"
              value={`${(result.original_probability * 100).toFixed(1)}%`}
            />
            <StatBox
              label="Simulated Risk"
              value={`${(result.counterfactual_probability * 100).toFixed(1)}%`}
              highlight
            />
            <StatBox
              label="Change"
              value={`${result.probability_delta > 0 ? "+" : ""}${(result.probability_delta * 100).toFixed(1)}%`}
              variant={result.probability_delta < 0 ? "success" : result.probability_delta > 0 ? "danger" : "neutral"}
            />
          </div>

          {/* Feature changes table */}
          {Object.keys(result.feature_changes).length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Changes Applied</h4>
              <div className="space-y-2">
                {Object.entries(result.feature_changes).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between bg-slate-50 rounded-lg px-4 py-2 text-sm">
                    <span className="text-slate-700 font-medium">{formatFeatureName(key)}</span>
                    <span className="font-mono text-slate-600">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="border-l-4 border-l-amber-400 bg-amber-50/60 rounded-r-lg px-4 py-3 text-xs text-slate-600 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
            <p>{result.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  );
};

function SliderCard({
  label,
  help,
  value,
  originalValue,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  help: string;
  value: number;
  originalValue: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  const delta = value - originalValue;
  const DeltaIcon = delta < 0 ? ArrowDown : delta > 0 ? ArrowUp : Minus;
  const deltaColor = delta < 0 ? "text-risk-low" : delta > 0 ? "text-risk-high" : "text-slate-400";

  return (
    <div className="card-padded">
      <div className="flex items-start justify-between mb-1">
        <div>
          <label className="block text-sm font-semibold text-slate-700">{label}</label>
          <p className="text-xs text-slate-400">{help}</p>
        </div>
        <div className="text-right">
          <span className="text-lg font-bold font-mono text-slate-800">
            {step < 1 ? value.toFixed(1) : value}
          </span>
          <span className="text-xs text-slate-500 ml-1">{unit}</span>
          {delta !== 0 && (
            <div className={`text-xs font-semibold flex items-center justify-end gap-0.5 ${deltaColor}`}>
              <DeltaIcon className="w-3 h-3" />
              {Math.abs(delta) < 1 ? Math.abs(delta).toFixed(1) : Math.round(Math.abs(delta))}
            </div>
          )}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-slate-200 rounded-lg accent-primary-600 cursor-pointer mt-2"
      />
    </div>
  );
}

function StatBox({
  label,
  value,
  highlight = false,
  variant = "neutral",
}: {
  label: string;
  value: string;
  highlight?: boolean;
  variant?: "success" | "danger" | "neutral";
}) {
  const variantStyles = {
    success: "text-risk-low",
    danger: "text-risk-high",
    neutral: "text-slate-800",
  };

  return (
    <div className={`rounded-xl p-4 text-center ${highlight ? "bg-primary-50 border border-primary-100" : "bg-slate-50"}`}>
      <p className="text-xs text-slate-500 font-medium mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${variantStyles[variant]}`}>{value}</p>
    </div>
  );
}
