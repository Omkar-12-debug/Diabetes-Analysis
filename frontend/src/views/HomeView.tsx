import React from "react";
import { Activity, ArrowRight } from "lucide-react";

interface HomeViewProps {
  onStart: () => void;
}

export const HomeView: React.FC<HomeViewProps> = ({ onStart }) => {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="card-padded text-center py-14 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-blue-50 opacity-60" />
        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary-100 text-primary-700 text-sm font-medium mb-6">
            <Activity className="w-4 h-4" />
            AI-Powered Clinical Screening
          </div>
          <h1 className="text-4xl font-extrabold text-slate-900 mb-8 leading-tight">
            Understand Your<br />
            <span className="text-primary-600">Diabetes Risk</span> Today
          </h1>
          <button
            onClick={onStart}
            className="btn-primary text-base px-8 py-3.5 inline-flex items-center gap-2 shadow-lg shadow-primary-200"
          >
            Check Your Risk Now
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Important Notice */}
      <div className="card-padded border-l-4 border-l-amber-400 bg-amber-50/60">
        <h3 className="font-semibold text-slate-800 mb-1.5 text-sm">
          ⚕️ Important Medical Notice
        </h3>
        <p className="text-sm text-slate-600 leading-relaxed">
          This tool is designed for <strong>educational screening purposes only</strong> and does
          not replace professional medical advice. Always consult your healthcare
          provider for clinical diagnosis and treatment decisions. The model predictions
          are based on statistical patterns and should be interpreted alongside
          comprehensive clinical evaluation.
        </p>
      </div>
    </div>
  );
};
