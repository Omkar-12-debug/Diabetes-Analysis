import React, { useState, useEffect, useCallback } from "react";
import { TrendingUp, Search, Activity, Clock, AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { AssessmentHistory } from "../types";
import { apiClient } from "../api/client";

interface HistoryViewProps {
  patientId: string | undefined;
}

export const HistoryView: React.FC<HistoryViewProps> = ({ patientId: initialId }) => {
  const [searchId, setSearchId] = useState(initialId || "");
  const [records, setRecords] = useState<AssessmentHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  const fetchHistory = useCallback(async (pid: string) => {
    if (!pid.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.getPatientHistory(pid.trim());
      setRecords(res.assessments || []);
      setSearched(true);
    } catch (err: any) {
      setError(err?.message || "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (initialId) fetchHistory(initialId);
  }, [initialId, fetchHistory]);

  const chartData = records
    .slice()
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((r) => ({
      date: new Date(r.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      risk: r.risk_score,
      category: r.risk_category,
    }));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-primary-600" />
          Visit History & Risk Trajectory
        </h2>
        <p className="text-sm text-slate-500">
          Track how your diabetes risk has changed over multiple assessments.
        </p>
      </div>

      {/* Search bar */}
      <div className="card-padded flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
        <div className="flex-1 relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && fetchHistory(searchId)}
            placeholder="Enter Patient ID (e.g. PT-HEALTHY-01)"
            className="input-field pl-9"
          />
        </div>
        <button
          onClick={() => fetchHistory(searchId)}
          disabled={loading || !searchId.trim()}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          Search
        </button>
      </div>

      {error && (
        <div className="card-padded border-l-4 border-l-red-400 bg-red-50/60 text-sm text-red-700">
          {error}
        </div>
      )}

      {searched && records.length === 0 && !loading && (
        <div className="card-padded text-center py-10">
          <AlertCircle className="w-10 h-10 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">
            No assessment records found for <strong>{searchId}</strong>.
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Run an assessment first, or try a different Patient ID.
          </p>
        </div>
      )}

      {records.length > 0 && (
        <>
          {/* Trajectory Chart */}
          <div className="card-padded">
            <h3 className="text-sm font-semibold text-slate-700 mb-4">Risk Score Over Time</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 12, fill: "#64748b" }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: "#64748b" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #e2e8f0",
                      borderRadius: "8px",
                      fontSize: "13px",
                    }}
                  />
                  <ReferenceLine y={30} stroke="#10b981" strokeDasharray="5 5" label={{ value: "Low / Moderate", fontSize: 10, fill: "#94a3b8" }} />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "Moderate / High", fontSize: 10, fill: "#94a3b8" }} />
                  <Line
                    type="monotone"
                    dataKey="risk"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    dot={{ fill: "#2563eb", r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Record list */}
          <div className="card-padded">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Assessment Records</h3>
            <div className="space-y-3">
              {records
                .slice()
                .reverse()
                .map((r) => {
                  const catColor =
                    r.risk_category === "Low"
                      ? "bg-risk-low-bg text-risk-low border-risk-low-border"
                      : r.risk_category === "Moderate"
                      ? "bg-risk-mod-bg text-risk-mod border-risk-mod-border"
                      : "bg-risk-high-bg text-risk-high border-risk-high-border";
                  return (
                    <div key={r.id} className="flex items-center justify-between bg-slate-50 rounded-lg px-4 py-3">
                      <div className="flex items-center gap-3">
                        <Clock className="w-4 h-4 text-slate-400" />
                        <div>
                          <p className="text-sm font-medium text-slate-700">
                            {new Date(r.created_at).toLocaleDateString("en-US", {
                              year: "numeric",
                              month: "long",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </p>
                          <p className="text-xs text-slate-400">
                            Model v{r.model_version}
                            {r.verified_diabetes_label !== null &&
                              r.verified_diabetes_label !== undefined &&
                              ` · Lab verified: ${r.verified_diabetes_label === 1 ? "Positive" : "Negative"}`}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-lg font-bold font-mono text-slate-800">{r.risk_score}%</span>
                        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${catColor}`}>
                          {r.risk_category}
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
