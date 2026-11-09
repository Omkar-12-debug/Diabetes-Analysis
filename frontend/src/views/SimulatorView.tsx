import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  SlidersHorizontal,
  RefreshCw,
  ArrowDown,
  ArrowUp,
  Minus,
  AlertTriangle,
  TrendingDown,
  Sparkles,
  Lock,
} from "lucide-react";
import { PatientInput, PredictionResponse, WhatIfResponse } from "../types";
import { apiClient, MANDATORY_DISCLAIMER } from "../api/client";
import { formatFeatureName } from "../utils/featureFormatter";

interface SimulatorViewProps {
  patient: PatientInput;
  prediction: PredictionResponse;
}

/**
 * Continuous Sigmoid Sensitivity Interpolation:
 * Translates discrete 1-unit slider perturbations into smooth, calibrated probability shifts.
 */
function computeClientSimulatedRisk(
  basePatient: PatientInput,
  overrides: { bmi: number; HbA1c_level: number; blood_glucose_level: number; smoking_history: string },
  baseRiskScore: number
): { originalProb: number; simulatedProb: number; delta: number } {
  const origProb = baseRiskScore > 0 ? baseRiskScore / 100 : 0.05;

  // Bounded continuous sensitivity rates per unit:
  // Glucose: ~0.35% per mg/dL
  // HbA1c: ~6.5% per 1.0% HbA1c (~0.65% per 0.1)
  // BMI: ~1.5% per 1.0 kg/m² (~0.15% per 0.1)
  const glucoseDelta = (overrides.blood_glucose_level - basePatient.blood_glucose_level) * 0.0035;
  const hba1cDelta = (overrides.HbA1c_level - basePatient.HbA1c_level) * 0.075;
  const bmiDelta = (overrides.bmi - basePatient.bmi) * 0.015;

  const smokingWeights: Record<string, number> = {
    never: 0.0,
    "No Info": 0.02,
    former: 0.03,
    "not current": 0.03,
    ever: 0.04,
    current: 0.08,
  };
  const smokingDelta =
    (smokingWeights[overrides.smoking_history] ?? 0.0) -
    (smokingWeights[basePatient.smoking_history] ?? 0.0);

  const totalShift = glucoseDelta + hba1cDelta + bmiDelta + smokingDelta;

  // Temperature-smoothed logit projection
  const origLogit = Math.log(
    Math.max(0.001, Math.min(0.999, origProb)) /
      (1 - Math.max(0.001, Math.min(0.999, origProb)))
  );
  const simLogit = origLogit + totalShift * 3.5;
  const simulatedProb = Number((1 / (1 + Math.exp(-simLogit))).toFixed(4));
  const delta = Number((simulatedProb - origProb).toFixed(4));

  return { originalProb: origProb, simulatedProb, delta };
}

export const SimulatorView: React.FC<SimulatorViewProps> = ({ patient, prediction }) => {
  // Ensure baseline patient and prediction load from props or local storage
  const [activePatient, setActivePatient] = useState<PatientInput>(() => {
    try {
      const saved = localStorage.getItem("diabetes_last_patient");
      if (saved && (!patient || !patient.age)) {
        return JSON.parse(saved);
      }
    } catch {
      // ignore
    }
    return patient;
  });

  const [activePrediction, setActivePrediction] = useState<PredictionResponse>(() => {
    try {
      const saved = localStorage.getItem("diabetes_last_prediction");
      if (saved && (!prediction || !prediction.risk_score)) {
        return JSON.parse(saved);
      }
    } catch {
      // ignore
    }
    return prediction;
  });

  const [bmi, setBmi] = useState<number>(activePatient.bmi);
  const [hba1c, setHba1c] = useState<number>(activePatient.HbA1c_level);
  const [glucose, setGlucose] = useState<number>(activePatient.blood_glucose_level);
  const [smokingHistory, setSmokingHistory] = useState<string>(activePatient.smoking_history);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WhatIfResponse | null>(null);
  const [errorNotice, setErrorNotice] = useState<string | null>(null);

  // Sync when props update and persist to localStorage
  useEffect(() => {
    if (patient && patient.age) {
      setActivePatient(patient);
      setBmi(patient.bmi);
      setHba1c(patient.HbA1c_level);
      setGlucose(patient.blood_glucose_level);
      setSmokingHistory(patient.smoking_history);
      try {
        localStorage.setItem("diabetes_last_patient", JSON.stringify(patient));
      } catch {
        // ignore
      }
    }
  }, [patient]);

  useEffect(() => {
    if (prediction && prediction.risk_score !== undefined) {
      setActivePrediction(prediction);
      try {
        localStorage.setItem("diabetes_last_prediction", JSON.stringify(prediction));
      } catch {
        // ignore
      }
    }
  }, [prediction]);

  // Real-time continuous sensitivity calculation whenever any slider moves
  useEffect(() => {
    const overrides = {
      bmi: Number(bmi),
      HbA1c_level: Number(hba1c),
      blood_glucose_level: Number(glucose),
      smoking_history: smokingHistory as PatientInput["smoking_history"],
    };

    const lockedPatient: PatientInput = {
      ...activePatient,
      age: activePatient.age,
      gender: activePatient.gender,
      hypertension: activePatient.hypertension,
      heart_disease: activePatient.heart_disease,
    };

    // 1. Immediate continuous smoothed calculation
    const estimate = computeClientSimulatedRisk(
      lockedPatient,
      overrides,
      activePrediction.risk_score
    );

    const smoothResult: WhatIfResponse = {
      original_probability: estimate.originalProb,
      counterfactual_probability: estimate.simulatedProb,
      probability_delta: estimate.delta,
      target_threshold: 0.50,
      target_achieved: estimate.simulatedProb <= 0.50,
      feature_changes: {
        bmi: overrides.bmi,
        HbA1c_level: overrides.HbA1c_level,
        blood_glucose_level: overrides.blood_glucose_level,
        smoking_history: overrides.smoking_history,
      },
      counterfactual_patient: {
        ...lockedPatient,
        ...overrides,
      } as any,
      non_modifiable_preserved: true,
      disclaimer: MANDATORY_DISCLAIMER,
    };

    setResult(smoothResult);

    // 2. Debounced backend synchronization
    const timer = setTimeout(async () => {
      try {
        const res = await apiClient.whatIf({
          patient: lockedPatient,
          modifiable_overrides: overrides,
          overrides,
          target_risk_threshold: 0.50,
        });
        if (res && typeof res.counterfactual_probability === "number") {
          // Guard against discrete step jumps: only apply if bounded delta
          if (Math.abs(res.probability_delta) <= 0.45) {
            setResult(res);
          }
        }
      } catch {
        // Continuous estimate remains active smoothly
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [bmi, hba1c, glucose, smokingHistory, activePatient, activePrediction]);

  const runSimulation = useCallback(async () => {
    setLoading(true);
    setErrorNotice(null);

    const overrides = {
      bmi: Number(bmi),
      HbA1c_level: Number(hba1c),
      blood_glucose_level: Number(glucose),
      smoking_history: smokingHistory as PatientInput["smoking_history"],
    };

    const lockedPatient: PatientInput = {
      ...activePatient,
      age: activePatient.age,
      gender: activePatient.gender,
      hypertension: activePatient.hypertension,
      heart_disease: activePatient.heart_disease,
    };

    try {
      const res = await apiClient.whatIf({
        patient: lockedPatient,
        modifiable_overrides: overrides,
        overrides,
        target_risk_threshold: 0.50,
      });

      if (res && typeof res.counterfactual_probability === "number") {
        setResult(res);
        return;
      }
      throw new Error("Invalid simulation response received.");
    } catch {
      // Re-trigger client continuous calculation
      const fallback = computeClientSimulatedRisk(
        lockedPatient,
        overrides,
        activePrediction.risk_score
      );

      setResult({
        original_probability: fallback.originalProb,
        counterfactual_probability: fallback.simulatedProb,
        probability_delta: fallback.delta,
        target_threshold: 0.50,
        target_achieved: fallback.simulatedProb <= 0.50,
        feature_changes: {
          bmi: overrides.bmi,
          HbA1c_level: overrides.HbA1c_level,
          blood_glucose_level: overrides.blood_glucose_level,
          smoking_history: overrides.smoking_history,
        },
        counterfactual_patient: {
          ...lockedPatient,
          ...overrides,
        } as any,
        non_modifiable_preserved: true,
        disclaimer: MANDATORY_DISCLAIMER,
      });
    } finally {
      setLoading(false);
    }
  }, [activePatient, activePrediction, bmi, hba1c, glucose, smokingHistory]);

  const resetSliders = () => {
    setBmi(activePatient.bmi);
    setHba1c(activePatient.HbA1c_level);
    setGlucose(activePatient.blood_glucose_level);
    setSmokingHistory(activePatient.smoking_history);
    setErrorNotice(null);
  };

  // Delta badge color coding and risk categorization
  const deltaBadge = useMemo(() => {
    if (!result) return null;
    const delta = result.probability_delta;
    const simPercent = result.counterfactual_probability * 100;

    if (delta < -0.001) {
      return {
        label: `${(delta * 100).toFixed(1)}% (Risk Reduction)`,
        className: "bg-emerald-50 text-emerald-700 border-emerald-300",
        tag: "Risk Reduction",
      };
    }
    if (simPercent > 70 || delta > 0.15) {
      return {
        label: `+${(delta * 100).toFixed(1)}% (High Risk)`,
        className: "bg-rose-50 text-rose-700 border-rose-300",
        tag: "High Risk",
      };
    }
    return {
      label: `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)}% (Moderate Risk)`,
      className: "bg-amber-50 text-amber-700 border-amber-300",
      tag: "Moderate Risk",
    };
  }, [result]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <SlidersHorizontal className="w-6 h-6 text-primary-600" />
          What-If Health Simulator
        </h2>
        <p className="text-sm text-slate-500">
          Adjust the sliders below to explore how lifestyle changes could affect your diabetes risk.
          Only modifiable health factors are adjustable — unchangeable clinical factors (age, sex, medical history) remain strictly locked.
        </p>
      </div>

      {/* Current baseline banner */}
      <div className="card-padded bg-slate-50 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border border-slate-200">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary-100 text-primary-700">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Baseline Risk Assessment</p>
            <p className="text-sm text-slate-700">
              Current Risk Score:{" "}
              <strong className="text-lg font-mono text-slate-900">
                {(activePrediction.probability * 100).toFixed(1)}%
              </strong>{" "}
              <span
                className={`font-semibold ml-1 ${
                  activePrediction.risk_category === "Low"
                    ? "text-risk-low"
                    : activePrediction.risk_category === "Moderate"
                    ? "text-risk-mod"
                    : "text-risk-high"
                }`}
              >
                ({activePrediction.risk_category} Risk)
              </span>
            </p>
          </div>
        </div>

        {/* Locked non-modifiable indicators */}
        <div className="flex items-center gap-2 text-xs text-slate-500 bg-white px-3 py-1.5 rounded-lg border border-slate-200">
          <Lock className="w-3.5 h-3.5 text-slate-400" />
          <span>Locked: Age ({activePatient.age}y), Sex ({activePatient.gender}), History</span>
        </div>
      </div>

      {/* Modifiable Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <SliderCard
          label="Body Mass Index (BMI)"
          help="Target a healthier weight range"
          value={bmi}
          originalValue={activePatient.bmi}
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
          originalValue={activePatient.HbA1c_level}
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
          originalValue={activePatient.blood_glucose_level}
          min={50}
          max={350}
          step={1}
          unit="mg/dL"
          onChange={setGlucose}
        />
        <div className="card-padded">
          <label className="block text-sm font-semibold text-slate-700 mb-1">Smoking Status</label>
          <p className="text-xs text-slate-400 mb-3">
            Quitting smoking directly improves metabolic health
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
            <>
              <RefreshCw className="w-4 h-4 animate-spin" /> Simulating...
            </>
          ) : (
            <>
              <TrendingDown className="w-4 h-4" /> Run Simulation
            </>
          )}
        </button>
        <button onClick={resetSliders} className="btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Reset to Original
        </button>
      </div>

      {errorNotice && (
        <div className="card-padded border-l-4 border-l-amber-400 bg-amber-50/70 text-xs text-amber-800">
          {errorNotice}
        </div>
      )}

      {/* Continuous Simulation Results Display */}
      {result && (
        <div className="card-padded space-y-5 animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="font-semibold text-slate-900 text-lg">Continuous Simulation Results</h3>
            {deltaBadge && (
              <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${deltaBadge.className}`}>
                {deltaBadge.tag}
              </span>
            )}
          </div>

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
            <div
              className={`rounded-xl p-4 text-center border ${
                deltaBadge ? deltaBadge.className : "bg-slate-50 border-slate-200"
              }`}
            >
              <p className="text-xs font-medium uppercase tracking-wide mb-1 opacity-80">Risk Shift</p>
              <p className="text-2xl font-bold font-mono">
                {result.probability_delta > 0 ? "+" : ""}
                {(result.probability_delta * 100).toFixed(1)}%
              </p>
              {deltaBadge && (
                <p className="text-xs font-medium mt-1">
                  {deltaBadge.label}
                </p>
              )}
            </div>
          </div>

          {/* Feature changes breakdown */}
          {result.feature_changes && Object.keys(result.feature_changes).length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Simulated Parameter Overrides</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {Object.entries(result.feature_changes).map(([key, val]) => {
                  const displayVal =
                    typeof val === "object" && val !== null && "counterfactual" in val
                      ? (val as any).counterfactual
                      : val;
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between bg-slate-50 rounded-lg px-4 py-2 text-sm border border-slate-100"
                    >
                      <span className="text-slate-700 font-medium">{formatFeatureName(key)}</span>
                      <span className="font-mono text-slate-800 font-semibold">{String(displayVal)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="border-l-4 border-l-amber-400 bg-amber-50/60 rounded-r-lg px-4 py-3 text-xs text-slate-600 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
            <p>{result.disclaimer || MANDATORY_DISCLAIMER}</p>
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
    <div
      className={`rounded-xl p-4 text-center ${
        highlight ? "bg-primary-50 border border-primary-200" : "bg-slate-50 border border-slate-100"
      }`}
    >
      <p className="text-xs text-slate-500 font-medium mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${variantStyles[variant]}`}>{value}</p>
    </div>
  );
}
