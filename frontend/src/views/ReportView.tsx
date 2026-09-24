import React, { useRef } from "react";
import {
  FileText,
  Printer,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { ExplanationResponse, PatientInput, PredictionResponse } from "../types";
import { formatFeatureName } from "../utils/featureFormatter";

interface ReportViewProps {
  patient: PatientInput;
  prediction: PredictionResponse;
  explanation: ExplanationResponse | null;
}

export const ReportView: React.FC<ReportViewProps> = ({ patient, prediction, explanation }) => {
  const reportRef = useRef<HTMLDivElement>(null);

  const handlePrint = () => {
    window.print();
  };

  const RiskIcon =
    prediction.risk_category === "Low"
      ? CheckCircle2
      : prediction.risk_category === "Moderate"
      ? AlertTriangle
      : ShieldAlert;

  const riskColorClass =
    prediction.risk_category === "Low"
      ? "text-risk-low"
      : prediction.risk_category === "Moderate"
      ? "text-risk-mod"
      : "text-risk-high";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
            <FileText className="w-6 h-6 text-primary-600" />
            Assessment Report
          </h2>
          <p className="text-sm text-slate-500">
            A printable summary you can share with your healthcare provider.
          </p>
        </div>
        <button onClick={handlePrint} className="btn-secondary flex items-center gap-2">
          <Printer className="w-4 h-4" />
          Print Report
        </button>
      </div>

      <div ref={reportRef} className="card-padded space-y-8 print:shadow-none print:border-none">
        {/* Header */}
        <div className="border-b border-slate-200 pb-5">
          <h3 className="text-lg font-bold text-slate-900">Diabetes Risk Assessment Report</h3>
          <p className="text-xs text-slate-400 mt-1">
            Generated on {new Date(prediction.timestamp).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
            {prediction.patient_id && ` · Patient ID: ${prediction.patient_id}`}
            {prediction.assessment_id && ` · Assessment #${prediction.assessment_id}`}
          </p>
        </div>

        {/* Risk Score */}
        <div className="text-center py-4">
          <RiskIcon className={`w-10 h-10 mx-auto mb-2 ${riskColorClass}`} />
          <p className={`text-4xl font-extrabold ${riskColorClass}`}>{prediction.risk_score}%</p>
          <p className={`text-sm font-semibold uppercase tracking-wide mt-1 ${riskColorClass}`}>
            {prediction.risk_category} Risk
          </p>
        </div>

        {/* Patient Profile */}
        <div>
          <h4 className="text-sm font-semibold text-slate-700 mb-3 border-b border-slate-100 pb-2">
            Patient Profile
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <InfoCell label="Age" value={`${patient.age} years`} />
            <InfoCell label="Biological Sex" value={patient.gender} />
            <InfoCell label="BMI" value={`${patient.bmi.toFixed(1)} kg/m²`} />
            <InfoCell label="HbA1c" value={`${patient.HbA1c_level.toFixed(1)}%`} />
            <InfoCell label="Blood Glucose" value={`${patient.blood_glucose_level} mg/dL`} />
            <InfoCell label="Smoking" value={patient.smoking_history} />
            <InfoCell label="Hypertension" value={patient.hypertension === 1 ? "Yes" : "No"} />
            <InfoCell label="Heart Disease" value={patient.heart_disease === 1 ? "Yes" : "No"} />
          </div>
        </div>

        {/* Key Factors */}
        {explanation && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-risk-high" /> Risk-Increasing Factors
              </h4>
              <ul className="space-y-1.5">
                {(explanation.top_risk_increasing_factors || []).map((f, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
                    {formatFeatureName(f.feature)}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                <TrendingDown className="w-4 h-4 text-risk-low" /> Protective Factors
              </h4>
              <ul className="space-y-1.5">
                {(explanation.top_risk_decreasing_factors || []).map((f, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
                    {formatFeatureName(f.feature)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="border-t border-slate-200 pt-4 text-xs text-slate-400 leading-relaxed">
          <strong>Disclaimer:</strong> This report is generated by an AI-powered screening tool
          and is intended for educational purposes only. It does not constitute a medical diagnosis.
          Please consult a qualified healthcare professional for clinical evaluation, treatment
          decisions, and follow-up care. Model: {prediction.model_name} v{prediction.model_version}.
        </div>
      </div>
    </div>
  );
};

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 rounded-lg px-3 py-2">
      <p className="text-xs text-slate-400 mb-0.5">{label}</p>
      <p className="text-sm font-medium text-slate-800">{value}</p>
    </div>
  );
}
