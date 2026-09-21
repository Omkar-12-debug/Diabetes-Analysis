import React from "react";
import { Sparkles, RefreshCw, HeartPulse, HelpCircle } from "lucide-react";
import { Gender, PatientInput, SmokingHistory } from "../types";

interface AssessmentViewProps {
  patient: PatientInput;
  onChange: (updated: PatientInput) => void;
  onSubmit: () => void;
  loading: boolean;
}

const PRESETS: Record<string, PatientInput> = {
  healthy: {
    patient_id: "PT-HEALTHY-01",
    gender: "Male",
    age: 28,
    hypertension: 0,
    heart_disease: 0,
    smoking_history: "never",
    bmi: 22.4,
    HbA1c_level: 5.1,
    blood_glucose_level: 85,
  },
  borderline: {
    patient_id: "PT-BORDER-02",
    gender: "Female",
    age: 48,
    hypertension: 1,
    heart_disease: 0,
    smoking_history: "former",
    bmi: 28.6,
    HbA1c_level: 6.2,
    blood_glucose_level: 135,
  },
  highRisk: {
    patient_id: "PT-HIGHRISK-03",
    gender: "Female",
    age: 62,
    hypertension: 1,
    heart_disease: 1,
    smoking_history: "current",
    bmi: 34.2,
    HbA1c_level: 7.8,
    blood_glucose_level: 195,
  },
};

export { PRESETS };

function SliderBlock({
  label,
  helpText,
  value,
  displayValue,
  min,
  max,
  step,
  colorFn,
  onChange,
}: {
  label: string;
  helpText: string;
  value: number;
  displayValue: string;
  min: number;
  max: number;
  step: number;
  colorFn: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="card-padded">
      <div className="flex items-start justify-between mb-3">
        <div>
          <label className="block text-sm font-semibold text-slate-700">{label}</label>
          <p className="text-xs text-slate-400 mt-0.5 flex items-center gap-1">
            <HelpCircle className="w-3 h-3" />
            {helpText}
          </p>
        </div>
        <span className={`text-lg font-bold font-mono ${colorFn(value)}`}>{displayValue}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-slate-200 rounded-lg accent-primary-600 cursor-pointer"
      />
    </div>
  );
}

export const AssessmentView: React.FC<AssessmentViewProps> = ({
  patient,
  onChange,
  onSubmit,
  loading,
}) => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1">
          Check Your Diabetes Risk
        </h2>
        <p className="text-sm text-slate-500">
          Fill in your health details below. All inputs are validated within safe clinical
          ranges and never stored externally.
        </p>
      </div>

      {/* Quick-fill presets */}
      <div className="card-padded">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-slate-600">
            <Sparkles className="w-4 h-4 text-amber-500" />
            Try a sample profile to explore instantly:
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => onChange(PRESETS.healthy)}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-green-50 text-risk-low border border-risk-low-border hover:bg-green-100 transition"
            >
              Healthy Adult
            </button>
            <button
              type="button"
              onClick={() => onChange(PRESETS.borderline)}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-amber-50 text-amber-700 border border-risk-mod-border hover:bg-amber-100 transition"
            >
              Borderline / Pre-Diabetic
            </button>
            <button
              type="button"
              onClick={() => onChange(PRESETS.highRisk)}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-red-50 text-risk-high border border-risk-high-border hover:bg-red-100 transition"
            >
              Elevated Risk
            </button>
          </div>
        </div>
      </div>

      <form
        onSubmit={(e) => { e.preventDefault(); onSubmit(); }}
        className="space-y-5"
      >
        {/* Patient ID */}
        <div className="card-padded">
          <label className="block text-sm font-semibold text-slate-700 mb-1">
            Patient Identifier (optional)
          </label>
          <p className="text-xs text-slate-400 mb-2">
            A reference ID for tracking your visits over time. One will be auto-generated if left blank.
          </p>
          <input
            type="text"
            value={patient.patient_id || ""}
            onChange={(e) => onChange({ ...patient, patient_id: e.target.value })}
            placeholder="e.g. PT-YOUR-NAME"
            className="input-field max-w-xs"
          />
        </div>

        {/* Demographics Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card-padded">
            <label className="block text-sm font-semibold text-slate-700 mb-1">
              Biological Sex
            </label>
            <select
              value={patient.gender}
              onChange={(e) => onChange({ ...patient, gender: e.target.value as Gender })}
              className="input-field"
            >
              <option value="Female">Female</option>
              <option value="Male">Male</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="card-padded">
            <label className="block text-sm font-semibold text-slate-700 mb-1">
              Smoking Status
            </label>
            <select
              value={patient.smoking_history}
              onChange={(e) => onChange({ ...patient, smoking_history: e.target.value as SmokingHistory })}
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

          <div className="card-padded">
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Medical History
            </label>
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={patient.hypertension === 1}
                  onChange={(e) => onChange({ ...patient, hypertension: e.target.checked ? 1 : 0 })}
                  className="w-4 h-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-slate-600">Diagnosed with high blood pressure</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={patient.heart_disease === 1}
                  onChange={(e) => onChange({ ...patient, heart_disease: e.target.checked ? 1 : 0 })}
                  className="w-4 h-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-slate-600">Diagnosed with heart disease</span>
              </label>
            </div>
          </div>
        </div>

        {/* Biometric Sliders */}
        <SliderBlock
          label="Age"
          helpText="Your current age in years"
          value={patient.age}
          displayValue={`${patient.age} years`}
          min={1}
          max={100}
          step={1}
          colorFn={() => "text-slate-800"}
          onChange={(v) => onChange({ ...patient, age: v })}
        />

        <SliderBlock
          label="Body Mass Index (BMI)"
          helpText="A measure of body weight relative to height (kg/m²)"
          value={patient.bmi}
          displayValue={`${patient.bmi.toFixed(1)} kg/m²`}
          min={12}
          max={65}
          step={0.1}
          colorFn={(v) => v < 25 ? "text-risk-low" : v < 30 ? "text-risk-mod" : "text-risk-high"}
          onChange={(v) => onChange({ ...patient, bmi: v })}
        />

        <SliderBlock
          label="HbA1c Level"
          helpText="Average blood sugar over the past 2–3 months (%)"
          value={patient.HbA1c_level}
          displayValue={`${patient.HbA1c_level.toFixed(1)}%`}
          min={3.5}
          max={15}
          step={0.1}
          colorFn={(v) => v < 5.7 ? "text-risk-low" : v < 6.5 ? "text-risk-mod" : "text-risk-high"}
          onChange={(v) => onChange({ ...patient, HbA1c_level: v })}
        />

        <SliderBlock
          label="Blood Glucose Level"
          helpText="Fasting blood sugar concentration (mg/dL)"
          value={patient.blood_glucose_level}
          displayValue={`${patient.blood_glucose_level} mg/dL`}
          min={50}
          max={350}
          step={1}
          colorFn={(v) => v < 100 ? "text-risk-low" : v <= 140 ? "text-risk-mod" : "text-risk-high"}
          onChange={(v) => onChange({ ...patient, blood_glucose_level: v })}
        />

        {/* Submit */}
        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full py-3.5 text-base flex items-center justify-center gap-2"
        >
          {loading ? (
            <><RefreshCw className="w-5 h-5 animate-spin" /> Analyzing your health data...</>
          ) : (
            <><HeartPulse className="w-5 h-5" /> Calculate My Risk</>
          )}
        </button>
      </form>
    </div>
  );
};
