import React, { useState, useEffect } from "react";
import { Sliders, Lock, ArrowRight, AlertOctagon, RotateCcw } from "lucide-react";
import { PatientInput, WhatIfResponse } from "../types";
import { apiClient, MANDATORY_DISCLAIMER } from "../api/client";

interface WhatIfSimulatorProps {
  basePatient: PatientInput;
}

export const WhatIfSimulator: React.FC<WhatIfSimulatorProps> = ({ basePatient }) => {
  // Modifiable override states
  const [bmi, setBmi] = useState<number>(basePatient.bmi);
  const [glucose, setGlucose] = useState<number>(basePatient.blood_glucose_level);
  const [hba1c, setHba1c] = useState<number>(basePatient.HbA1c_level);
  const [smoking, setSmoking] = useState<string>(basePatient.smoking_history);

  const [simulation, setSimulation] = useState<WhatIfResponse | null>(null);
  const [simulating, setSimulating] = useState<boolean>(false);

  // Sync state when basePatient changes
  useEffect(() => {
    setBmi(basePatient.bmi);
    setGlucose(basePatient.blood_glucose_level);
    setHba1c(basePatient.HbA1c_level);
    setSmoking(basePatient.smoking_history);
  }, [basePatient]);

  // Run what-if query
  const runSimulation = async () => {
    setSimulating(true);
    try {
      const res = await apiClient.whatIf({
        patient: basePatient,
        modifiable_overrides: {
          bmi,
          blood_glucose_level: glucose,
          HbA1c_level: hba1c,
          smoking_history: smoking as any,
        },
      });
      setSimulation(res);
    } catch {
      // Handled in client
    } finally {
      setSimulating(false);
    }
  };

  // Run automatically when values change (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      runSimulation();
    }, 250);
    return () => clearTimeout(timer);
  }, [bmi, glucose, hba1c, smoking, basePatient]);

  const resetToBaseline = () => {
    setBmi(basePatient.bmi);
    setGlucose(basePatient.blood_glucose_level);
    setHba1c(basePatient.HbA1c_level);
    setSmoking(basePatient.smoking_history);
  };

  const delta = simulation?.probability_delta ?? 0;
  const isReduced = delta < 0;

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-clinical-border">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 mb-5 border-b border-clinical-border/80 gap-2">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-clinical-cyan" />
            Counterfactual "What-If" Sensitivity Simulator
          </h3>
          <p className="text-xs text-slate-400">
            Simulate hypothetical interventions on clinically modifiable risk factors
          </p>
        </div>

        <button
          type="button"
          onClick={resetToBaseline}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset to Baseline
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Modifiable Factor Controls */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span>Modifiable Variables:</span>
            <span className="text-[11px] text-slate-500 font-normal">Interactive Sliders</span>
          </div>

          {/* BMI Slider */}
          <div className="p-3.5 rounded-xl bg-clinical-card/50 border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-medium text-slate-300">Target BMI</span>
              <span className="text-xs font-mono font-bold text-clinical-cyan">
                {bmi.toFixed(1)} kg/m²{" "}
                <span className="text-slate-400 font-normal">
                  (Base: {basePatient.bmi.toFixed(1)})
                </span>
              </span>
            </div>
            <input
              type="range"
              min="15"
              max="50"
              step="0.2"
              value={bmi}
              onChange={(e) => setBmi(Number(e.target.value))}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
          </div>

          {/* Glucose Slider */}
          <div className="p-3.5 rounded-xl bg-clinical-card/50 border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-medium text-slate-300">Target Fasting Blood Glucose</span>
              <span className="text-xs font-mono font-bold text-clinical-cyan">
                {glucose} mg/dL{" "}
                <span className="text-slate-400 font-normal">
                  (Base: {basePatient.blood_glucose_level})
                </span>
              </span>
            </div>
            <input
              type="range"
              min="60"
              max="260"
              step="1"
              value={glucose}
              onChange={(e) => setGlucose(Number(e.target.value))}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
          </div>

          {/* HbA1c Slider */}
          <div className="p-3.5 rounded-xl bg-clinical-card/50 border border-clinical-border/50">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-medium text-slate-300">Target HbA1c Level</span>
              <span className="text-xs font-mono font-bold text-clinical-cyan">
                {hba1c.toFixed(1)}%{" "}
                <span className="text-slate-400 font-normal">
                  (Base: {basePatient.HbA1c_level.toFixed(1)})
                </span>
              </span>
            </div>
            <input
              type="range"
              min="4.0"
              max="12.0"
              step="0.1"
              value={hba1c}
              onChange={(e) => setHba1c(Number(e.target.value))}
              className="w-full accent-clinical-cyan h-1.5 bg-slate-700 rounded-lg cursor-pointer"
            />
          </div>

          {/* Non-modifiable Locked Badges */}
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
            <div className="flex items-center gap-1.5 text-slate-400 font-semibold mb-2">
              <Lock className="w-3.5 h-3.5 text-slate-500" /> Non-Modifiable Clinical Factors (Locked):
            </div>
            <div className="flex flex-wrap gap-2 text-[11px] font-mono text-slate-400">
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                Age: {basePatient.age} yrs
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                Sex: {basePatient.gender}
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                Hypertension: {basePatient.hypertension ? "Yes" : "No"}
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                Heart Disease: {basePatient.heart_disease ? "Yes" : "No"}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Simulation Outcome & Delta */}
        <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
          <div className="p-4 rounded-xl glass-card border border-clinical-border flex flex-col items-center text-center">
            <div className="flex items-center justify-between w-full mb-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Simulated Probability Shift (Δp)
              </span>
              {simulating && (
                <span className="text-[10px] font-mono text-clinical-cyan animate-pulse">
                  Recalculating...
                </span>
              )}
            </div>

            <div className="flex items-center justify-center gap-4 my-2">
              <div className="flex flex-col items-center">
                <span className="text-xs text-slate-400">Baseline</span>
                <span className="text-xl font-bold font-mono text-slate-300">
                  {((simulation?.original_probability ?? 0.5) * 100).toFixed(1)}%
                </span>
              </div>

              <ArrowRight className="w-5 h-5 text-slate-500" />

              <div className="flex flex-col items-center">
                <span className="text-xs text-slate-400">Counterfactual</span>
                <span
                  className={`text-xl font-bold font-mono ${
                    isReduced ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {((simulation?.counterfactual_probability ?? 0.5) * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Delta pill */}
            <div
              className={`mt-2 px-3 py-1 rounded-full text-xs font-mono font-extrabold flex items-center gap-1.5 ${
                isReduced
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
              }`}
            >
              <span>Net Risk Delta:</span>
              <span>{delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`}</span>
            </div>
          </div>

          {/* Mandatory Non-Causal Medical Disclaimer */}
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
            <div className="flex items-center gap-2 font-bold mb-1">
              <AlertOctagon className="w-4 h-4 text-amber-400 shrink-0" />
              <span>Mandatory Clinical Sensitivity Disclaimer</span>
            </div>
            <p className="text-[11px] text-amber-200/90 leading-relaxed font-sans">
              {MANDATORY_DISCLAIMER}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
