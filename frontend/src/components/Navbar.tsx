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
    <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        {/* Top brand row */}
        <div className="flex items-center justify-between py-3">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-50 border border-primary-100">
              <Activity className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 leading-tight">
                DiabetesPredict
              </h1>
              <p className="text-xs text-slate-500">
                Early detection for better health
              </p>
            </div>
          </div>

          <button
            onClick={() => onTabChange("assess")}
            className="btn-primary text-sm hidden sm:flex items-center gap-2"
          >
            <ClipboardList className="w-4 h-4" />
            Check Your Risk
          </button>
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
                    ? "border-primary-600 text-primary-700"
                    : isDisabled
                    ? "border-transparent text-slate-300 cursor-not-allowed"
                    : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                }`}
              >
                {tab.icon}
                <span className="hidden md:inline">{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
