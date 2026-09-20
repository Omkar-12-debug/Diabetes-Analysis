import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { History, TrendingUp, CheckCircle2 } from "lucide-react";
import { AssessmentHistory } from "../types";
import { apiClient } from "../api/client";

interface PatientHistoryProps {
  patientId: string;
  history: AssessmentHistory[];
  onRefresh: () => void;
}

export const PatientHistory: React.FC<PatientHistoryProps> = ({
  patientId,
  history,
  onRefresh,
}) => {
  const [feedbackAssessmentId, setFeedbackAssessmentId] = useState<number | null>(null);
  const [feedbackLabel, setFeedbackLabel] = useState<0 | 1>(1);
  const [feedbackNotes, setFeedbackNotes] = useState<string>("");
  const [submittingFeedback, setSubmittingFeedback] = useState<boolean>(false);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  const handleGroundTruthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!feedbackAssessmentId) return;

    setSubmittingFeedback(true);
    setFeedbackMessage(null);
    try {
      const res = await apiClient.recordGroundTruth({
        assessment_id: feedbackAssessmentId,
        verified_diabetes_label: feedbackLabel,
        notes: feedbackNotes,
      });
      setFeedbackMessage(res.message);
      setFeedbackNotes("");
      onRefresh();
    } catch {
      setFeedbackMessage("Failed to submit ground truth label.");
    } finally {
      setSubmittingFeedback(false);
    }
  };

  // Format data for Recharts line chart
  const chartData = history.map((item, index) => ({
    visitIndex: `Visit ${index + 1}`,
    score: item.risk_score,
    probability: Number((item.risk_probability * 100).toFixed(1)),
    category: item.risk_category,
    date: item.created_at ? new Date(item.created_at).toLocaleTimeString() : `Visit ${index + 1}`,
    assessmentId: item.id,
    verified: item.verified_diabetes_label,
  }));

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-clinical-border">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 mb-5 border-b border-clinical-border/80 gap-2">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <History className="w-4 h-4 text-clinical-cyan" />
            Longitudinal Patient Risk Trajectory
          </h3>
          <p className="text-xs text-slate-400">
            Consecutive assessments and risk progression for patient ID:{" "}
            <span className="font-mono text-clinical-cyan font-semibold">{patientId}</span>
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="px-2.5 py-1 rounded-lg bg-clinical-card border border-clinical-border font-mono text-slate-300">
            Total Visits: {history.length}
          </span>
        </div>
      </div>

      {history.length === 0 ? (
        <div className="p-8 text-center text-slate-400 flex flex-col items-center">
          <TrendingUp className="w-8 h-8 text-slate-600 mb-2" />
          <p className="text-sm">No historical trajectory recorded for this patient yet.</p>
          <p className="text-xs text-slate-500 mt-1">
            Running assessments will automatically plot longitudinal visits here.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Trajectory Line Chart */}
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="visitIndex" stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#131b2e",
                    borderColor: "#263554",
                    borderRadius: "0.75rem",
                    fontSize: "12px",
                    color: "#e2e8f0",
                  }}
                  formatter={(value: any) => [`${value} / 100`, "Risk Score"]}
                />
                {/* Reference Risk Tier Thresholds */}
                <ReferenceLine y={70} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: "High (70)", fill: "#f43f5e", fontSize: 10 }} />
                <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" label={{ value: "Low (30)", fill: "#10b981", fontSize: 10 }} />

                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#06b6d4"
                  strokeWidth={3}
                  dot={{ r: 5, fill: "#06b6d4", stroke: "#0a0f1d", strokeWidth: 2 }}
                  activeDot={{ r: 7, fill: "#6366f1" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Assessment Visit Badges with Ground Truth Feedback Form */}
          <div className="pt-2 border-t border-clinical-border/60">
            <h4 className="text-xs font-semibold text-slate-300 mb-3 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-clinical-cyan" />
              Recorded Visits & Delayed Clinical Diagnostic Feedback
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-4">
              {history.map((h, i) => (
                <div
                  key={h.id || i}
                  onClick={() => setFeedbackAssessmentId(h.id)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    feedbackAssessmentId === h.id
                      ? "bg-clinical-cyan/10 border-clinical-cyan shadow-md"
                      : "bg-clinical-card/60 border-clinical-border hover:border-slate-600"
                  }`}
                >
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-bold text-white">Visit #{i + 1}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        h.risk_category === "Low"
                          ? "bg-emerald-500/20 text-emerald-400"
                          : h.risk_category === "Moderate"
                          ? "bg-amber-500/20 text-amber-400"
                          : "bg-rose-500/20 text-rose-400"
                      }`}
                    >
                      {h.risk_category} ({h.risk_score})
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Glucose: {h.input_features?.blood_glucose_level ?? "N/A"} | HbA1c: {h.input_features?.HbA1c_level ?? "N/A"}%
                  </div>
                  {h.verified_diabetes_label !== null && h.verified_diabetes_label !== undefined && (
                    <div className="mt-1 text-[10px] font-bold text-clinical-cyan">
                      Verified Diagnosis: {h.verified_diabetes_label === 1 ? "Diabetic (1)" : "Non-Diabetic (0)"}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Ingestion Form for Delayed Clinical Ground Truth */}
            {feedbackAssessmentId && (
              <form
                onSubmit={handleGroundTruthSubmit}
                className="p-4 rounded-xl bg-clinical-card border border-clinical-border space-y-3"
              >
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-white">
                    Submit Confirmed Diagnostic Outcome for Assessment #{feedbackAssessmentId}
                  </span>
                  <button
                    type="button"
                    onClick={() => setFeedbackAssessmentId(null)}
                    className="text-slate-400 hover:text-white"
                  >
                    Cancel
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] text-slate-300 mb-1">
                      Confirmed Diagnostic Result
                    </label>
                    <select
                      value={feedbackLabel}
                      onChange={(e) => setFeedbackLabel(Number(e.target.value) as 0 | 1)}
                      className="w-full bg-clinical-dark border border-clinical-border rounded-lg px-2.5 py-1.5 text-xs text-white"
                    >
                      <option value={1}>Diabetic (1) — Positive Diagnosis</option>
                      <option value={0}>Non-Diabetic (0) — Negative Diagnosis</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] text-slate-300 mb-1">
                      Clinical Confirmation Notes (Optional)
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. Fasting plasma glucose confirmed >= 126 mg/dL"
                      value={feedbackNotes}
                      onChange={(e) => setFeedbackNotes(e.target.value)}
                      className="w-full bg-clinical-dark border border-clinical-border rounded-lg px-2.5 py-1.5 text-xs text-white"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-1">
                  {feedbackMessage && (
                    <span className="text-xs text-emerald-400 font-medium">
                      {feedbackMessage}
                    </span>
                  )}
                  <button
                    type="submit"
                    disabled={submittingFeedback}
                    className="ml-auto px-4 py-1.5 rounded-lg bg-clinical-cyan hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
                  >
                    {submittingFeedback ? "Submitting..." : "Save Ground Truth"}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
