import React from "react";
import { Activity, Cpu, ShieldCheck } from "lucide-react";
import { HealthResponse, ModelInfoResponse } from "../types";

interface HeaderProps {
  health?: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  isLive: boolean;
}

export const Header: React.FC<HeaderProps> = ({ modelInfo, isLive }) => {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-clinical-border px-6 py-4 shadow-xl">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & Title */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-tr from-clinical-cyan/20 to-clinical-indigo/30 border border-clinical-cyan/40 shadow-inner">
            <Activity className="w-6 h-6 text-clinical-cyan animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                DiaGuard AI
              </h1>
              <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-clinical-indigo/20 text-clinical-indigo border border-clinical-indigo/40 tracking-wide uppercase">
                Clinical CDSS
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Calibrated Risk Prediction, SHAP Explainability & Counterfactual Simulation
            </p>
          </div>
        </div>

        {/* System & Model Badges */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Active Model Alias */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass-card text-xs">
            <Cpu className="w-3.5 h-3.5 text-clinical-cyan" />
            <span className="text-slate-400">Model:</span>
            <span className="font-semibold text-slate-200 font-mono">
              {modelInfo?.model_name || "DiabetesRiskModel"}
            </span>
            <span className="px-1.5 py-0.2 rounded bg-clinical-cyan/20 text-clinical-cyan font-bold text-[10px]">
              @{modelInfo?.alias || "champion"}
            </span>
            <span className="text-slate-500 font-mono text-[11px]">v{modelInfo?.model_version || "2"}</span>
          </div>

          {/* Quality Gate Status */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass-card text-xs">
            <ShieldCheck className="w-3.5 h-3.5 text-clinical-low" />
            <span className="text-slate-400">Gate:</span>
            <span className="font-semibold text-clinical-low">
              {modelInfo?.status === "promoted_by_quality_gate" ? "PASSED" : "VERIFIED"}
            </span>
          </div>

          {/* Live API / Fallback Indicator */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass-card text-xs">
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  isLive ? "bg-clinical-low" : "bg-clinical-mod"
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  isLive ? "bg-clinical-low" : "bg-clinical-mod"
                }`}
              />
            </span>
            <span className="text-slate-400">Service:</span>
            <span className={`font-semibold ${isLive ? "text-clinical-low" : "text-clinical-mod"}`}>
              {isLive ? "Live API (8000)" : "Fallback Simulation"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
