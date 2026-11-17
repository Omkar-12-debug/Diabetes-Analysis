import React from "react";
import {
  Activity,
  ArrowRight,
  ShieldCheck,
  Cpu,
  SlidersHorizontal,
  BarChart2,
  Gauge,
  LineChart,
} from "lucide-react";

interface HomeViewProps {
  onStart: () => void;
}

export const HomeView: React.FC<HomeViewProps> = ({ onStart }) => {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="card-padded text-center py-14 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-blue-50 opacity-60" />
        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary-100 text-primary-700 text-sm font-medium mb-6">
            <Activity className="w-4 h-4" />
            AI-Powered Clinical Screening &amp; MLOps
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-6 leading-tight">
            Understand Your<br />
            <span className="text-primary-600">Diabetes Risk</span> with Explainable AI
          </h1>
          <p className="max-w-2xl mx-auto text-base text-slate-600 mb-8 leading-relaxed">
            A clinical decision support platform combining calibrated machine learning,
            individualized SHAP feature attributions, and actionable counterfactual simulations.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onStart}
              className="btn-primary text-base px-8 py-3.5 inline-flex items-center gap-2 shadow-lg shadow-primary-200"
            >
              Start Clinical Assessment
              <ArrowRight className="w-5 h-5" />
            </button>
            <a
              href="http://localhost:3000"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary text-base px-6 py-3.5 inline-flex items-center gap-2"
            >
              <Gauge className="w-4 h-4 text-amber-500" />
              View MLOps Telemetry
            </a>
          </div>
        </div>
      </div>

      {/* 4 Feature Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="card-padded bg-white hover:border-primary-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 mb-1.5 text-base">Calibrated ML Model</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Production XGBoost model tuned for 10.3:1 imbalance, delivering calibrated 0–100 risk scores (PR-AUC: 0.886, Recall: 90.6%).
          </p>
        </div>

        <div className="card-padded bg-white hover:border-primary-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center mb-4">
            <BarChart2 className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 mb-1.5 text-base">TreeSHAP Explanations</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Transparency at the patient level: decomposes each risk score into positive and negative clinical biometric drivers.
          </p>
        </div>

        <div className="card-padded bg-white hover:border-primary-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center mb-4">
            <SlidersHorizontal className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 mb-1.5 text-base">What-If Counterfactuals</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Actionable simulations: explore realistic lifestyle modifications to safely lower risk while locking immutable traits like age.
          </p>
        </div>

        <div className="card-padded bg-white hover:border-primary-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center mb-4">
            <LineChart className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 mb-1.5 text-base">Telemetry &amp; Auditing</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Full enterprise lifecycle: Prometheus latency tracking, Grafana observability, and delayed ground truth verification.
          </p>
        </div>
      </div>

      {/* Important Notice */}
      <div className="card-padded border-l-4 border-l-amber-400 bg-amber-50/60">
        <h3 className="font-semibold text-slate-800 mb-1.5 text-sm flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-amber-600" />
          Clinical Advisory &amp; Notice
        </h3>
        <p className="text-sm text-slate-600 leading-relaxed">
          This system is intended for clinical screening, educational exploration, and MLOps operational demonstration.
          It does not substitute diagnostic clinical pathology. All counterfactual simulations represent hypothetical model sensitivities.
        </p>
      </div>
    </div>
  );
};
