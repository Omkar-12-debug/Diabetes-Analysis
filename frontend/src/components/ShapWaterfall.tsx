import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { Layers, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { ExplanationResponse } from "../types";

interface ShapWaterfallProps {
  explanation: ExplanationResponse | null;
  loading: boolean;
}

const FEATURE_LABEL_MAP: Record<string, string> = {
  blood_glucose_level: "Blood Glucose",
  HbA1c_level: "HbA1c Level",
  glucose_hba1c_interaction: "Glucose × HbA1c",
  bmi: "Body Mass Index",
  age: "Patient Age",
  age_bmi_interaction: "Age × BMI",
  cardiometabolic_risk: "Cardiometabolic Risk",
  hypertension: "Hypertension",
  heart_disease: "Heart Disease",
  smoking_history: "Smoking History",
  gender: "Biological Sex",
};

export const ShapWaterfall: React.FC<ShapWaterfallProps> = ({ explanation, loading }) => {
  if (loading) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center min-h-[300px] text-slate-400 animate-pulse">
        <p className="text-sm font-medium">Computing Local SHAP Feature Attributions...</p>
      </div>
    );
  }

  if (!explanation || !explanation.attributions || explanation.attributions.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center min-h-[300px] text-slate-400">
        <Layers className="w-10 h-10 text-slate-600 mb-2" />
        <p className="text-sm font-medium">No explanation available</p>
      </div>
    );
  }

  // Format data for horizontal bar chart (sort by absolute impact, top 6)
  const chartData = [...explanation.attributions]
    .sort((a, b) => Math.abs(b.attribution_value) - Math.abs(a.attribution_value))
    .slice(0, 7)
    .map((item) => ({
      featureName: FEATURE_LABEL_MAP[item.feature] || item.feature.replace(/_/g, " "),
      rawFeature: item.feature,
      value: Number(item.attribution_value.toFixed(3)),
      direction: item.direction,
      raw_value: item.raw_value,
    }))
    .reverse(); // Reverse for clean top-to-bottom display in Recharts

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-clinical-border">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 mb-4 border-b border-clinical-border/80 gap-2">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-clinical-cyan" />
            Explainable AI: Local SHAP Feature Attributions
          </h3>
          <p className="text-xs text-slate-400">
            Additive contribution to log-odds margin relative to base rate E[f(x)] = {explanation.base_value.toFixed(2)}
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 text-rose-400 font-semibold">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            Increases Risk (+)
          </div>
          <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            Lowers Risk (-)
          </div>
        </div>
      </div>

      {/* Top Driver Quick Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20">
          <div className="flex items-center gap-1.5 text-xs font-bold text-rose-400 mb-1">
            <ArrowUpRight className="w-4 h-4" /> Primary Risk Accelerators:
          </div>
          <div className="space-y-1">
            {explanation.top_risk_increasing_factors.slice(0, 2).map((f, i) => (
              <div key={i} className="flex justify-between text-xs text-slate-300">
                <span>{FEATURE_LABEL_MAP[f.feature] || f.feature}:</span>
                <span className="font-mono text-rose-400 font-bold">+{f.attribution_value.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
          <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-400 mb-1">
            <ArrowDownRight className="w-4 h-4" /> Primary Protective Factors:
          </div>
          <div className="space-y-1">
            {explanation.top_risk_decreasing_factors.slice(0, 2).map((f, i) => (
              <div key={i} className="flex justify-between text-xs text-slate-300">
                <span>{FEATURE_LABEL_MAP[f.feature] || f.feature}:</span>
                <span className="font-mono text-emerald-400 font-bold">{f.attribution_value.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recharts Horizontal Bar Attribution */}
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 20, left: 30, bottom: 5 }}
          >
            <XAxis
              type="number"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              domain={["auto", "auto"]}
            />
            <YAxis
              type="category"
              dataKey="featureName"
              stroke="#94a3b8"
              fontSize={12}
              tickLine={false}
              width={130}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#131b2e",
                borderColor: "#263554",
                borderRadius: "0.75rem",
                fontSize: "12px",
                color: "#e2e8f0",
              }}
              formatter={(value: any) => [`${value > 0 ? "+" : ""}${value}`, "SHAP Attribution"]}
            />
            <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
            <Bar dataKey="value" radius={[4, 4, 4, 4]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.value >= 0 ? "#f43f5e" : "#10b981"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
