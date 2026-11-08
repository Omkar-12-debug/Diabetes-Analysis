import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  TrendingUp,
  Search,
  Activity,
  Clock,
  AlertCircle,
  Database,
  User,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
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
  const [patientRecords, setPatientRecords] = useState<AssessmentHistory[]>([]);
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientError, setPatientError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  const [globalRecords, setGlobalRecords] = useState<AssessmentHistory[]>([]);
  const [globalLoading, setGlobalLoading] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const trajectoryRef = useRef<HTMLDivElement>(null);

  const fetchPatientHistory = useCallback(async (pid: string) => {
    if (!pid.trim()) return;
    setPatientLoading(true);
    setPatientError(null);
    try {
      const res = await apiClient.getPatientHistory(pid.trim());
      setPatientRecords(res.assessments || []);
      setSearched(true);
    } catch (err: any) {
      setPatientError(err?.message || "Failed to load patient history.");
    } finally {
      setPatientLoading(false);
    }
  }, []);

  const fetchGlobalHistory = useCallback(async () => {
    setGlobalLoading(true);
    setGlobalError(null);
    try {
      const res = await apiClient.getGlobalHistory(100, 0);
      setGlobalRecords(res.assessments || []);
    } catch (err: any) {
      setGlobalError(err?.message || "Failed to load global assessment ledger.");
    } finally {
      setGlobalLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGlobalHistory();
  }, [fetchGlobalHistory]);

  useEffect(() => {
    if (initialId) {
      setSearchId(initialId);
      fetchPatientHistory(initialId);
    }
  }, [initialId, fetchPatientHistory]);

  const handleSelectPatient = (pid: string) => {
    setSearchId(pid);
    fetchPatientHistory(pid);
    trajectoryRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const chartData = patientRecords
    .slice()
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((r) => ({
      date: new Date(r.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      risk: r.risk_score,
      category: r.risk_category,
    }));

  const getCategoryBadgeClass = (category: string) => {
    switch (category) {
      case "Low":
        return "bg-risk-low-bg text-risk-low border-risk-low-border";
      case "Moderate":
        return "bg-risk-mod-bg text-risk-mod border-risk-mod-border";
      case "High":
        return "bg-risk-high-bg text-risk-high border-risk-high-border";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-primary-600" />
          Clinical Assessment Ledger & Risk Trajectory
        </h2>
        <p className="text-sm text-slate-500">
          Review longitudinal progression for specific patients or explore the global registry of all recent assessments.
        </p>
      </div>

      {/* SECTION 1: Patient Longitudinal Trajectory */}
      <div ref={trajectoryRef} className="card-padded space-y-6 scroll-mt-20 border-primary-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <User className="w-5 h-5 text-primary-600" />
              Patient Longitudinal Trajectory
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Enter a Patient ID or click "View Trajectory" from the global ledger below.
            </p>
          </div>
          {searchId && (
            <span className="text-xs font-mono font-medium bg-slate-100 text-slate-700 px-2.5 py-1 rounded-md self-start sm:self-auto">
              Active ID: {searchId}
            </span>
          )}
        </div>

        {/* Search bar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <div className="flex-1 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchId}
              onChange={(e) => setSearchId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchPatientHistory(searchId)}
              placeholder="Enter Patient ID (e.g. PT-HEALTHY-01)"
              className="input-field pl-9"
            />
          </div>
          <button
            onClick={() => fetchPatientHistory(searchId)}
            disabled={patientLoading || !searchId.trim()}
            className="btn-primary flex items-center justify-center gap-2"
          >
            {patientLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            Search Trajectory
          </button>
        </div>

        {patientError && (
          <div className="p-4 border-l-4 border-l-red-400 bg-red-50/60 text-sm text-red-700 rounded-r-lg">
            {patientError}
          </div>
        )}

        {searched && patientRecords.length === 0 && !patientLoading && (
          <div className="text-center py-8 bg-slate-50 rounded-xl border border-dashed border-slate-200">
            <AlertCircle className="w-9 h-9 text-slate-300 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-600">
              No historical assessments found for <strong>{searchId}</strong>.
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Run an initial screening under the "Check Risk" tab or select another patient below.
            </p>
          </div>
        )}

        {patientRecords.length > 0 && (
          <div className="space-y-6">
            {/* Trajectory Chart */}
            <div className="bg-slate-50/80 p-4 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-slate-700">
                  Risk Trajectory ({patientRecords.length} assessment{patientRecords.length > 1 ? "s" : ""})
                </h4>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Low (&lt;30)
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Moderate (30-70)
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> High (&gt;70)
                  </span>
                </div>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 12, fill: "#64748b" }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: "#64748b" }} unit="%" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "white",
                        border: "1px solid #e2e8f0",
                        borderRadius: "8px",
                        fontSize: "13px",
                      }}
                      formatter={(val: any) => [`${val}%`, "Risk Score"]}
                    />
                    <ReferenceLine y={30} stroke="#10b981" strokeDasharray="5 5" />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="5 5" />
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

            {/* Patient Record Cards */}
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-3">Assessment Timeline</h4>
              <div className="space-y-2.5">
                {patientRecords
                  .slice()
                  .reverse()
                  .map((r) => {
                    const badgeClass = getCategoryBadgeClass(r.risk_category);
                    const bio = r.input_features || {};
                    return (
                      <div
                        key={r.id}
                        className="flex flex-col sm:flex-row sm:items-center justify-between bg-slate-50 hover:bg-slate-100/70 transition-colors rounded-lg px-4 py-3 gap-3 border border-slate-100"
                      >
                        <div className="flex items-start sm:items-center gap-3">
                          <Clock className="w-4 h-4 text-slate-400 mt-0.5 sm:mt-0 shrink-0" />
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="text-sm font-medium text-slate-800">
                                {new Date(r.created_at).toLocaleDateString("en-US", {
                                  year: "numeric",
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </p>
                              <span className="text-xs text-slate-400 font-mono">#{r.id}</span>
                            </div>
                            <p className="text-xs text-slate-500 mt-0.5">
                              Glucose: <span className="font-medium text-slate-700">{bio.blood_glucose_level ?? "—"} mg/dL</span> ·
                              HbA1c: <span className="font-medium text-slate-700">{bio.HbA1c_level ?? "—"}%</span> ·
                              BMI: <span className="font-medium text-slate-700">{bio.bmi ? Number(bio.bmi).toFixed(1) : "—"}</span>
                              {r.verified_diabetes_label !== null && r.verified_diabetes_label !== undefined && (
                                <span className="ml-2 font-semibold text-primary-700 bg-primary-50 px-1.5 py-0.5 rounded">
                                  Lab: {r.verified_diabetes_label === 1 ? "Diabetic" : "Non-Diabetic"}
                                </span>
                              )}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 self-end sm:self-auto">
                          <span className="text-lg font-bold font-mono text-slate-800">{r.risk_score}%</span>
                          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${badgeClass}`}>
                            {r.risk_category}
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 2: Global Assessment History */}
      <div className="card-padded space-y-5">
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <Database className="w-5 h-5 text-primary-600" />
              Global Assessment History
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Live audit ledger of recent assessments across all screened patients.
            </p>
          </div>
          <button
            onClick={fetchGlobalHistory}
            disabled={globalLoading}
            className="btn-secondary text-xs flex items-center gap-1.5 py-1.5 px-3"
            title="Refresh Ledger"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${globalLoading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {globalError && (
          <div className="p-4 border-l-4 border-l-red-400 bg-red-50/60 text-sm text-red-700 rounded-r-lg">
            {globalError}
          </div>
        )}

        {globalLoading && globalRecords.length === 0 ? (
          <div className="text-center py-12 text-slate-400 flex flex-col items-center justify-center gap-2">
            <Activity className="w-6 h-6 animate-spin text-primary-600" />
            <p className="text-sm">Loading global assessment ledger...</p>
          </div>
        ) : globalRecords.length === 0 ? (
          <div className="text-center py-10 bg-slate-50 rounded-xl border border-dashed border-slate-200">
            <Database className="w-9 h-9 text-slate-300 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-600">No assessments logged in the system yet.</p>
            <p className="text-xs text-slate-400 mt-1">Run an assessment in the "Check Risk" tab to populate this ledger.</p>
          </div>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-slate-600 text-xs font-semibold uppercase tracking-wider border-b border-slate-200">
                <tr>
                  <th scope="col" className="px-4 py-3">Timestamp</th>
                  <th scope="col" className="px-4 py-3">Patient ID</th>
                  <th scope="col" className="px-4 py-3">Biometrics</th>
                  <th scope="col" className="px-4 py-3">Risk Score</th>
                  <th scope="col" className="px-4 py-3">Category</th>
                  <th scope="col" className="px-4 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {globalRecords.map((r) => {
                  const bio = r.input_features || {};
                  const badgeClass = getCategoryBadgeClass(r.risk_category);
                  const isCurrent = r.patient_id === searchId;
                  return (
                    <tr
                      key={r.id}
                      className={`hover:bg-slate-50/80 transition-colors ${isCurrent ? "bg-primary-50/40" : ""}`}
                    >
                      <td className="px-4 py-3 whitespace-nowrap text-xs text-slate-500 font-mono">
                        {new Date(r.created_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap font-medium text-slate-800">
                        {r.patient_id}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-xs text-slate-600">
                        <span className="inline-block mr-2">Age: <strong>{bio.age ?? "—"}</strong></span>
                        <span className="inline-block mr-2">BMI: <strong>{bio.bmi ? Number(bio.bmi).toFixed(1) : "—"}</strong></span>
                        <span className="inline-block mr-2">Glu: <strong>{bio.blood_glucose_level ?? "—"}</strong></span>
                        <span className="inline-block">HbA1c: <strong>{bio.HbA1c_level ?? "—"}%</strong></span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap font-bold font-mono text-slate-800">
                        {r.risk_score}%
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${badgeClass}`}>
                          {r.risk_category}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-right">
                        <button
                          onClick={() => handleSelectPatient(r.patient_id)}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-primary-600 hover:text-primary-700 bg-primary-50 hover:bg-primary-100 px-2.5 py-1.5 rounded-lg transition-colors"
                        >
                          View Trajectory
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
