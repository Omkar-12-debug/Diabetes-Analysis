import React from "react";
import { AlertTriangle, CheckCircle, ShieldAlert, Timer } from "lucide-react";
import { PredictionResponse, RiskCategory } from "../types";

interface RiskGaugeProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ prediction, loading }) => {
  if (loading) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center min-h-[340px] text-slate-400 animate-pulse">
        <div className="w-36 h-36 rounded-full border-4 border-dashed border-slate-700 animate-spin mb-4" />
        <p className="text-sm font-medium">Computing Calibrated Model Inference...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center min-h-[340px] text-slate-400">
        <ShieldAlert className="w-12 h-12 text-slate-600 mb-3" />
        <p className="text-sm font-medium">No assessment run yet</p>
        <p className="text-xs text-slate-500 mt-1">Configure clinical parameters and click "Compute Risk"</p>
      </div>
    );
  }

  const { risk_score, probability, risk_category, model_version } = prediction;

  // Arc calculation for SVG circular meter
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  // Use a 270-degree arc gauge
  const strokeDashoffset = circumference - (circumference * (risk_score / 100) * 0.75);

  const getTierDetails = (cat: RiskCategory) => {
    switch (cat) {
      case "Low":
        return {
          color: "#10b981",
          textColor: "text-emerald-400",
          bgColor: "bg-emerald-500/10",
          borderColor: "border-emerald-500/30",
          icon: <CheckCircle className="w-5 h-5 text-emerald-400" />,
          label: "Low Risk Tier",
          subtext: "Clinical risk score is within healthy baseline bounds (<30/100).",
        };
      case "Moderate":
        return {
          color: "#f59e0b",
          textColor: "text-amber-400",
          bgColor: "bg-amber-500/10",
          borderColor: "border-amber-500/30",
          icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
          label: "Moderate Risk Tier",
          subtext: "Sub-clinical glucose or metabolic factors elevated (30–70/100). Lifestyle intervention recommended.",
        };
      case "High":
        return {
          color: "#f43f5e",
          textColor: "text-rose-400",
          bgColor: "bg-rose-500/10",
          borderColor: "border-rose-500/30",
          icon: <ShieldAlert className="w-5 h-5 text-rose-400" />,
          label: "High Risk Tier",
          subtext: "Elevated risk markers indicate high probability of onset (>70/100). Further diagnostic evaluation advised.",
        };
    }
  };

  const tier = getTierDetails(risk_category);

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-clinical-border flex flex-col items-center text-center relative overflow-hidden">
      {/* Background radial glow based on risk tier */}
      <div
        className="absolute top-1/3 w-48 h-48 rounded-full blur-3xl opacity-20 pointer-events-none transition-all duration-700"
        style={{ background: tier.color }}
      />

      <div className="w-full flex items-center justify-between pb-3 mb-2 border-b border-clinical-border/80 text-xs">
        <span className="font-semibold text-slate-300">Clinical Risk Stratification</span>
        <div className="flex items-center gap-1.5 text-slate-400 font-mono">
          <Timer className="w-3.5 h-3.5 text-clinical-cyan" />
          <span>Champion v{model_version}</span>
        </div>
      </div>

      {/* SVG Circular / Arched Gauge */}
      <div className="relative my-3 flex items-center justify-center">
        <svg className="w-48 h-48 transform -rotate-135" viewBox="0 0 160 160">
          {/* Background Track */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke="#1e293b"
            strokeWidth="12"
            fill="transparent"
            strokeDasharray={circumference * 0.75}
            strokeLinecap="round"
          />
          {/* Animated Value Stroke */}
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke={tier.color}
            strokeWidth="12"
            fill="transparent"
            strokeDasharray={circumference * 0.75}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        {/* Center Content */}
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-4xl font-black font-mono tracking-tight text-white">
            {risk_score}
          </span>
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">
            Score / 100
          </span>
          <span className="text-[10px] font-mono text-slate-500 mt-0.5">
            P = {(probability * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Risk Tier Badge & Details */}
      <div className={`mt-2 w-full p-3.5 rounded-xl border ${tier.bgColor} ${tier.borderColor} transition-all`}>
        <div className="flex items-center justify-center gap-2 mb-1">
          {tier.icon}
          <span className={`text-sm font-bold ${tier.textColor}`}>{tier.label}</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed max-w-sm mx-auto">
          {tier.subtext}
        </p>
      </div>
    </div>
  );
};
