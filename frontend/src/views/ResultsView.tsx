import React from "react";
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Info,
  SlidersHorizontal,
  ArrowRight,
} from "lucide-react";
import { ExplanationResponse, PredictionResponse } from "../types";
import { formatFeatureName } from "../utils/featureFormatter";

interface ResultsViewProps {
  prediction: PredictionResponse;
  explanation: ExplanationResponse | null;
  onSimulate: () => void;
}

function RiskBadge({ category, score }: { category: string; score: number }) {
  const config: Record<string, { bg: string; text: string; ring: string; icon: React.ReactNode }> = {
    Low: {
      bg: "bg-risk-low-bg",
      text: "text-risk-low",
      ring: "ring-risk-low-border",
      icon: <CheckCircle2 className="w-8 h-8" />,
    },
    Moderate: {
      bg: "bg-risk-mod-bg",
      text: "text-risk-mod",
      ring: "ring-risk-mod-border",
      icon: <AlertTriangle className="w-8 h-8" />,
    },
    High: {
      bg: "bg-risk-high-bg",
      text: "text-risk-high",
      ring: "ring-risk-high-border",
      icon: <ShieldAlert className="w-8 h-8" />,
    },
  };
  const c = config[category] || config.Moderate;

  return (
    <div className={`inline-flex flex-col items-center gap-2 px-8 py-6 rounded-2xl ring-2 ${c.bg} ${c.text} ${c.ring}`}>
      {c.icon}
      <span className="text-4xl font-extrabold">{score}</span>
      <span className="text-sm font-semibold uppercase tracking-wide">{category} Risk</span>
    </div>
  );
}

function riskGuidance(category: string): string {
  switch (category) {
    case "Low":
      return "Your current health indicators suggest a lower likelihood of diabetes. Keep up healthy habits — regular exercise, a balanced diet, and routine health checkups will help you stay on track.";
    case "Moderate":
      return "Your profile shows some risk factors worth discussing with your doctor. Consider scheduling a checkup, especially if you haven't had blood work done recently. Small lifestyle changes can make a real difference.";
    case "High":
      return "Several of your health indicators are in a range that warrants prompt medical attention. Please consult your healthcare provider soon for a comprehensive evaluation and personalized guidance.";
    default:
      return "";
  }
}

export const ResultsView: React.FC<ResultsViewProps> = ({ prediction, explanation, onSimulate }) => {
  const riskFactors = explanation?.top_risk_increasing_factors || [];
  const protectiveFactors = explanation?.top_risk_decreasing_factors || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1">Your Assessment Results</h2>
        <p className="text-sm text-slate-500">
          Assessed on {new Date(prediction.timestamp).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
        </p>
      </div>

      {/* Risk Score Hero Card */}
      <div className="card-padded text-center py-10">
        <p className="text-sm font-medium text-slate-500 mb-4">
          Based on your health profile, your estimated diabetes risk score is:
        </p>
        <RiskBadge category={prediction.risk_category} score={prediction.risk_score} />
        <p className="text-base text-slate-600 mt-6 max-w-xl mx-auto leading-relaxed">
          {riskGuidance(prediction.risk_category)}
        </p>
      </div>

      {/* What's Driving Your Score */}
      {explanation && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Risk-increasing factors */}
          <div className="card-padded border-l-4 border-l-red-400">
            <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-risk-high" />
              Factors Increasing Your Risk
            </h3>
            {riskFactors.length > 0 ? (
              <ul className="space-y-3">
                {riskFactors.map((f, i) => (
                  <li key={i} className="flex items-start gap-3 text-sm">
                    <div className="w-2 h-2 rounded-full bg-red-400 mt-1.5 shrink-0" />
                    <div>
                      <span className="font-medium text-slate-800">
                        {formatFeatureName(f.feature)}
                      </span>
                      <p className="text-slate-500 text-xs mt-0.5">
                        This factor is contributing to a higher risk estimate.
                        {f.raw_value !== null && f.raw_value !== undefined &&
                          ` (Your value: ${f.raw_value})`}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-400 italic">No significant risk-increasing factors identified.</p>
            )}
          </div>

          {/* Risk-decreasing factors */}
          <div className="card-padded border-l-4 border-l-green-400">
            <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-risk-low" />
              Factors Working in Your Favor
            </h3>
            {protectiveFactors.length > 0 ? (
              <ul className="space-y-3">
                {protectiveFactors.map((f, i) => (
                  <li key={i} className="flex items-start gap-3 text-sm">
                    <div className="w-2 h-2 rounded-full bg-green-400 mt-1.5 shrink-0" />
                    <div>
                      <span className="font-medium text-slate-800">
                        {formatFeatureName(f.feature)}
                      </span>
                      <p className="text-slate-500 text-xs mt-0.5">
                        This factor is helping lower your risk estimate.
                        {f.raw_value !== null && f.raw_value !== undefined &&
                          ` (Your value: ${f.raw_value})`}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-400 italic">No significant protective factors identified.</p>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="card-padded flex flex-col sm:flex-row items-center gap-4 justify-between">
        <div className="flex items-start gap-2 text-sm text-slate-500">
          <Info className="w-4 h-4 mt-0.5 shrink-0" />
          <span>
            Want to see how lifestyle changes might affect your risk? Try the What-If Simulator.
          </span>
        </div>
        <button onClick={onSimulate} className="btn-primary flex items-center gap-2 whitespace-nowrap">
          <SlidersHorizontal className="w-4 h-4" />
          Open Simulator
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
