import React from "react";
import { Sparkles, User, HeartPulse, Stethoscope, RefreshCw } from "lucide-react";
import { Gender, PatientInput, SmokingHistory } from "../types";

interface AssessmentFormProps {
  patient: PatientInput;
  onChange: (updated: PatientInput) => void;
  onSubmit: () => void;
  loading: boolean;
}

export const PRESETS: Record<string, PatientInput> = {
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
    patient_id: "PT-DIABETIC-03",
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

export const AssessmentForm: React.FC<AssessmentFormProps> = ({
  patient,
  onChange,
  onSubmit,
  loading,
}) => {
  const handlePreset = (key: keyof typeof PRESETS) => {
    onChange(PRESETS[key]);
  };

  const generateNewId = () => {
    const randomHex = Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0");
    onChange({ ...patient, patient_id: `PT-${randomHex.toUpperCase()}` });
  };

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-clinical-border">
      {/* Header & Preset Buttons */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-clinical-border/80">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-clinical-cyan" />
            Patient Clinical Metrics
          </h2>
          <p className="text-xs text-slate-400">
            Validated against physiological clinical bounds
          </p>
        </div>

        {/* Presets */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-medium flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Presets:
          </span>
          <button
            type="button"
            onClick={() => handlePreset("healthy")}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 transition-all"
          >
            Healthy
          </button>
          <button
            type="button"
            onClick={() => handlePreset("borderline")}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-all"
          >
            Borderline
          </button>
          <button
            type="button"
            onClick={() => handlePreset("highRisk")}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-all"
          >
            High Risk
          </button>
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="space-y-5"
      >
        {/* Patient ID */}
        <div className="flex items-center justify-between gap-3 bg-clinical-card/50 p-3 rounded-xl border border-clinical-border/50">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <User className="w-4 h-4 text-clinical-cyan" />
            <span>Patient ID:</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={patient.patient_id || ""}
              onChange={(e) => onChange({ ...patient, patient_id: e.target.value })}
              placeholder="e.g. PT-1002"
              className="bg-clinical-dark border border-clinical-border rounded-lg px-3 py-1 text-xs font-mono text-white focus:outline-none focus:border-clinical-cyan w-36"
            />
            <button
              type="button"
              onClick={generateNewId}
              title="Generate new patient UUID"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Demographics & History Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Gender */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Biological Sex / Gender
            </label>
            <select
              value={patient.gender}
              onChange={(e) => onChange({ ...patient, gender: e.target.value as Gender })}
              className="w-full bg-clinical-card border border-clinical-border rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-clinical-cyan"
            >
              <option value="Female">Female</option>
              <option value="Male">Male</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Smoking History */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Smoking History
            </label>
            <select
              value={patient.smoking_history}
              onChange={(e) =>
                onChange({ ...patient, smoking_history: e.target.value as SmokingHistory })
              }
              className="w-full bg-clinical-card border border-clinical-border rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-clinical-cyan"
            >
              <option value="never">Never</option>
              <option value="former">Former</option>
              <option value="current">Current</option>
              <option value="not current">Not Current</option>
              <option value="ever">Ever</option>
              <option value="No Info">No Info</option>
            </select>
          </div>

          {/* Comorbidities */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Cardiovascular Comorbidities
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() =>
                  onChange({ ...patient, hypertension: (patient.hypertension === 1 ? 0 : 1) as any })
                }
                className={`flex-1 py-2 px-2 text-xs font-semibold rounded-xl border transition-all ${
                  patient.hypertension === 1
                    ? "bg-rose-500/20 border-rose-500/50 text-rose-300"
                    : "bg-clinical-card border-clinical-border text-slate-400 hover:text-slate-200"
                }`}
              >
                Hypertension: {patient.hypertension === 1 ? "Yes" : "No"}
              </button>
              <button
                type="button"
                onClick={() =>
                  onChange({ ...patient, heart_disease: (patient.heart_disease === 1 ? 0 : 1) as any })
                }
                className={`flex-1 py-2 px-2 text-xs font-semibold rounded-xl border transition-all ${
                  patient.heart_disease === 1
                    ? "bg-rose-500/20 border-rose-500/50 text-rose-300"
                    : "bg-clinical-card border-clinical-border text-slate-400 hover:text-slate-200"
                }`}
              >
                Heart Disease: {patient.heart_disease === 1 ? "Yes" : "No"}
              </button>
            </div>
          </div>
        </div>

        {/* Quantitative Biometrics */}
        <div className="space-y-4 pt-2">
          {/* Age Slider */}
          <div className="bg-clinical-card/40 p-3.5 rounded-xl border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-semibold text-slate-300">Age (Years)</span>
              <span className="text-xs font-mono font-bold text-clinical-cyan">
                {patient.age} yrs
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={patient.age}
              onChange={(e) => onChange({ ...patient, age: Number(e.target.value) })}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>0 yrs</span>
              <span>Pediatric / Adult / Geriatric</span>
              <span>100 yrs</span>
            </div>
          </div>

          {/* BMI Slider */}
          <div className="bg-clinical-card/40 p-3.5 rounded-xl border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-semibold text-slate-300">
                Body Mass Index (BMI in kg/m²)
              </span>
              <span
                className={`text-xs font-mono font-bold ${
                  patient.bmi < 25
                    ? "text-emerald-400"
                    : patient.bmi < 30
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}
              >
                {patient.bmi.toFixed(1)} kg/m²
              </span>
            </div>
            <input
              type="range"
              min="12"
              max="65"
              step="0.1"
              value={patient.bmi}
              onChange={(e) => onChange({ ...patient, bmi: Number(e.target.value) })}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>Underweight (&lt;18.5)</span>
              <span>Normal (18.5–24.9)</span>
              <span>Overweight (25–29.9)</span>
              <span>Obese (≥30)</span>
            </div>
          </div>

          {/* HbA1c Slider */}
          <div className="bg-clinical-card/40 p-3.5 rounded-xl border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-semibold text-slate-300">
                HbA1c Glycated Hemoglobin (%)
              </span>
              <span
                className={`text-xs font-mono font-bold ${
                  patient.HbA1c_level < 5.7
                    ? "text-emerald-400"
                    : patient.HbA1c_level < 6.5
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}
              >
                {patient.HbA1c_level.toFixed(1)}%
              </span>
            </div>
            <input
              type="range"
              min="3.5"
              max="15.0"
              step="0.1"
              value={patient.HbA1c_level}
              onChange={(e) => onChange({ ...patient, HbA1c_level: Number(e.target.value) })}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>Normal (&lt;5.7%)</span>
              <span>Pre-diabetes (5.7–6.4%)</span>
              <span>Diabetes (≥6.5%)</span>
            </div>
          </div>

          {/* Blood Glucose Level */}
          <div className="bg-clinical-card/40 p-3.5 rounded-xl border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-semibold text-slate-300">
                Blood Glucose Concentration (mg/dL)
              </span>
              <span
                className={`text-xs font-mono font-bold ${
                  patient.blood_glucose_level < 100
                    ? "text-emerald-400"
                    : patient.blood_glucose_level <= 140
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}
              >
                {patient.blood_glucose_level} mg/dL
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="350"
              step="1"
              value={patient.blood_glucose_level}
              onChange={(e) =>
                onChange({ ...patient, blood_glucose_level: Number(e.target.value) })
              }
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>Normal (&lt;100)</span>
              <span>Impaired (100–140)</span>
              <span>Hyperglycemia (&gt;140)</span>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3.5 px-6 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-clinical-cyan to-clinical-indigo hover:from-cyan-400 hover:to-indigo-500 shadow-lg shadow-cyan-500/20 active:scale-[0.99] transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Evaluating Clinical Risk...
            </>
          ) : (
            <>
              <HeartPulse className="w-4 h-4" />
              Compute Calibrated Risk Assessment
            </>
          )}
        </button>
      </form>
    </div>
  );
};
