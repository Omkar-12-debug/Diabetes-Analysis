import React, { useState } from "react";
import { FileCheck, CheckCircle2, RefreshCw, AlertCircle } from "lucide-react";
import { apiClient } from "../api/client";

interface FeedbackViewProps {
  assessmentId: number | undefined;
  patientId: string | undefined;
}

export const FeedbackView: React.FC<FeedbackViewProps> = ({ assessmentId, patientId: _patientId }) => {
  const [assId, setAssId] = useState(assessmentId ? String(assessmentId) : "");
  const [label, setLabel] = useState<0 | 1 | null>(null);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!assId || label === null) return;
    setLoading(true);
    setError(null);
    try {
      await apiClient.recordGroundTruth({
        assessment_id: Number(assId),
        verified_diabetes_label: label,
        notes: notes || undefined,
      });
      setSubmitted(true);
    } catch (err: any) {
      setError(err?.message || "Submission failed.");
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div className="max-w-md mx-auto text-center card-padded py-14 space-y-4">
        <CheckCircle2 className="w-14 h-14 text-risk-low mx-auto" />
        <h2 className="text-xl font-bold text-slate-900">Thank You!</h2>
        <p className="text-sm text-slate-500">
          Your lab feedback has been recorded. This helps us monitor model accuracy
          and improve future predictions.
        </p>
        <button
          onClick={() => {
            setSubmitted(false);
            setAssId("");
            setLabel(null);
            setNotes("");
          }}
          className="btn-secondary inline-flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Submit Another
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <FileCheck className="w-6 h-6 text-primary-600" />
          Lab Result Feedback
        </h2>
        <p className="text-sm text-slate-500">
          If you've received a confirmed diagnosis from your doctor, you can record it here to
          help improve the model's accuracy over time.
        </p>
      </div>

      <div className="card-padded space-y-5">
        {/* Assessment ID */}
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-1">
            Assessment ID
          </label>
          <p className="text-xs text-slate-400 mb-2">
            The numeric ID from your original assessment (shown in your results).
          </p>
          <input
            type="number"
            value={assId}
            onChange={(e) => setAssId(e.target.value)}
            placeholder="e.g. 12345"
            className="input-field max-w-xs"
          />
        </div>

        {/* Clinical diagnosis */}
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Confirmed Lab Diagnosis
          </label>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setLabel(1)}
              className={`flex-1 py-3 rounded-lg border-2 text-sm font-semibold transition-all ${
                label === 1
                  ? "border-risk-high bg-risk-high-bg text-risk-high"
                  : "border-slate-200 text-slate-500 hover:border-slate-300"
              }`}
            >
              Diabetes Confirmed
            </button>
            <button
              type="button"
              onClick={() => setLabel(0)}
              className={`flex-1 py-3 rounded-lg border-2 text-sm font-semibold transition-all ${
                label === 0
                  ? "border-risk-low bg-risk-low-bg text-risk-low"
                  : "border-slate-200 text-slate-500 hover:border-slate-300"
              }`}
            >
              No Diabetes
            </button>
          </div>
        </div>

        {/* Notes */}
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-1">
            Additional Notes (optional)
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="e.g. Confirmed by fasting glucose test on 09/2026"
            rows={3}
            className="input-field resize-none"
          />
        </div>

        {error && (
          <div className="flex items-start gap-2 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            {error}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={loading || !assId || label === null}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? (
            <><RefreshCw className="w-4 h-4 animate-spin" /> Submitting...</>
          ) : (
            <><FileCheck className="w-4 h-4" /> Record Lab Result</>
          )}
        </button>
      </div>

      <div className="card-padded border-l-4 border-l-blue-400 bg-blue-50/60 text-xs text-slate-600">
        <strong>Privacy note:</strong> Your feedback is stored locally and used solely to
        evaluate model accuracy. It does not affect your current risk assessment or
        any medical records.
      </div>
    </div>
  );
};
