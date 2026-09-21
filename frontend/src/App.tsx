import { useState, useCallback } from "react";
import { Navbar, TabId } from "./components/Navbar";
import { HomeView } from "./views/HomeView";
import { AssessmentView, PRESETS } from "./views/AssessmentView";
import { ResultsView } from "./views/ResultsView";
import { SimulatorView } from "./views/SimulatorView";
import { HistoryView } from "./views/HistoryView";
import { FeedbackView } from "./views/FeedbackView";
import { ReportView } from "./views/ReportView";
import { ExplanationResponse, PatientInput, PredictionResponse } from "./types";
import { apiClient } from "./api/client";

const DEFAULT_PATIENT: PatientInput = { ...PRESETS.healthy };

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("home");
  const [patient, setPatient] = useState<PatientInput>(DEFAULT_PATIENT);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const hasResults = prediction !== null;

  const handleTabChange = useCallback((tab: TabId) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    try {
      const [pred, expl] = await Promise.all([
        apiClient.predict(patient),
        apiClient.explain(patient),
      ]);
      setPrediction(pred);
      setExplanation(expl);
      setActiveTab("results");
    } catch (err) {
      console.error("Prediction/Explanation failed:", err);
    } finally {
      setLoading(false);
    }
  }, [patient]);

  const renderView = () => {
    switch (activeTab) {
      case "home":
        return <HomeView onStart={() => handleTabChange("assess")} />;
      case "assess":
        return (
          <AssessmentView
            patient={patient}
            onChange={setPatient}
            onSubmit={handleSubmit}
            loading={loading}
          />
        );
      case "results":
        return prediction ? (
          <ResultsView
            prediction={prediction}
            explanation={explanation}
            onSimulate={() => handleTabChange("simulator")}
          />
        ) : null;
      case "simulator":
        return prediction ? (
          <SimulatorView patient={patient} prediction={prediction} />
        ) : null;
      case "history":
        return <HistoryView patientId={prediction?.patient_id} />;
      case "feedback":
        return (
          <FeedbackView
            assessmentId={prediction?.assessment_id}
            patientId={prediction?.patient_id}
          />
        );
      case "report":
        return prediction ? (
          <ReportView
            patient={patient}
            prediction={prediction}
            explanation={explanation}
          />
        ) : null;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        hasResults={hasResults}
      />
      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        {renderView()}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white mt-12 py-6">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 text-center">
          <p className="text-xs text-slate-400">
            DiabetesPredict · Educational AI screening tool · Not a medical diagnosis ·{" "}
            {new Date().getFullYear()}
          </p>
        </div>
      </footer>
    </div>
  );
}
