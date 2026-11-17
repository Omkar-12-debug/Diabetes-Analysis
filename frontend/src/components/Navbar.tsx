import React from "react";
import {
  Home,
  ClipboardList,
  BarChart3,
  SlidersHorizontal,
  TrendingUp,
  FileCheck,
  FileText,
  Activity,
  Cpu,
  ExternalLink,
  Gauge,
  Database,
} from "lucide-react";

export type TabId =
  | "home"
  | "assess"
  | "results"
  | "simulator"
  | "history"
  | "feedback"
  | "report";

interface NavbarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  hasResults: boolean;
}

const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "home", label: "Home", icon: <Home className="w-4 h-4" /> },
  { id: "assess", label: "Check Risk", icon: <ClipboardList className="w-4 h-4" /> },
  { id: "results", label: "Results & Insights", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "simulator", label: "What-If Simulator", icon: <SlidersHorizontal className="w-4 h-4" /> },
  { id: "history", label: "Visit History", icon: <TrendingUp className="w-4 h-4" /> },
  { id: "feedback", label: "Lab Feedback", icon: <FileCheck className="w-4 h-4" /> },
  { id: "report", label: "Assessment Report", icon: <FileText className="w-4 h-4" /> },
];

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, hasResults }) => {
  return (
    <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50 print:hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        {/* Top brand & MLOps row */}
        <div className="flex flex-wrap items-center justify-between py-3 gap-3">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-50 border border-primary-100">
              <Activity className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-slate-900 leading-tight">
                  DiabetesPredict AI
                </h1>
                <span className="hidden sm:inline-block text-[10px] font-semibold uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full">
                  MLOps Live
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Calibrated Screening · TreeSHAP · DiCE What-If
              </p>
            </div>
          </div>

          {/* Center Model Governance Badge (Desktop) */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
            <Cpu className="w-4 h-4 text-primary-600" />
            <span className="text-slate-500">Champion Model:</span>
            <span className="font-semibold text-slate-800 font-mono">XGBoost</span>
            <span className="bg-primary-100 text-primary-700 px-1.5 py-0.5 rounded font-mono font-bold text-[10px]">
              @champion v2
            </span>
            <span className="text-slate-300">|</span>
            <span className="text-slate-600">
              PR-AUC: <strong className="text-slate-800 font-mono">0.886</strong>
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-slate-600">
              Recall: <strong className="text-emerald-700 font-mono">90.6%</strong>
            </span>
          </div>

          {/* Right Action & MLOps Quick-Launch Links */}
          <div className="flex items-center gap-2">
            {/* Quick Links to Telemetry */}
            <div className="hidden md:flex items-center gap-1.5 bg-slate-100/80 p-1 rounded-lg border border-slate-200 text-xs font-medium">
              <a
                href="http://localhost:3000"
                target="_blank"
                rel="noopener noreferrer"
                title="Open Grafana Real-Time Observability Dashboard"
                className="flex items-center gap-1 px-2 py-1 rounded hover:bg-white hover:text-primary-700 text-slate-600 transition-colors"
              >
                <Gauge className="w-3.5 h-3.5 text-amber-500" />
                <span>Grafana</span>
                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
              </a>

              <a
                href="http://localhost:9090/targets"
                target="_blank"
                rel="noopener noreferrer"
                title="Open Prometheus Scrape Targets"
                className="flex items-center gap-1 px-2 py-1 rounded hover:bg-white hover:text-primary-700 text-slate-600 transition-colors"
              >
                <Activity className="w-3.5 h-3.5 text-orange-500" />
                <span>Prometheus</span>
                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
              </a>

              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                title="Open FastAPI Swagger Interactive Playground"
                className="flex items-center gap-1 px-2 py-1 rounded hover:bg-white hover:text-primary-700 text-slate-600 transition-colors"
              >
                <Database className="w-3.5 h-3.5 text-emerald-600" />
                <span>API Docs</span>
                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
              </a>
            </div>

            <button
              onClick={() => onTabChange("assess")}
              className="btn-primary text-sm flex items-center gap-2"
            >
              <ClipboardList className="w-4 h-4" />
              <span>Check Risk</span>
            </button>
          </div>
        </div>

        {/* Tab navigation row */}
        <nav className="-mb-px flex gap-1 overflow-x-auto pb-0">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            const isDisabled =
              (tab.id === "results" || tab.id === "simulator" || tab.id === "report") &&
              !hasResults;

            return (
              <button
                key={tab.id}
                onClick={() => !isDisabled && onTabChange(tab.id)}
                disabled={isDisabled}
                className={`flex items-center gap-1.5 px-3 py-2.5 text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
                  isActive
                    ? "border-primary-600 text-primary-700 font-semibold"
                    : isDisabled
                    ? "border-transparent text-slate-300 cursor-not-allowed"
                    : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
