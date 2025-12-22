"use client";

import { useState, useEffect } from "react";
import { usePredictionContext } from "./prediction-context";
import { Button } from "@/components/ui/button";
import PredictionForm from "./PredictionForm";
import ResultsDashboard from "./ResultsDashboard";
import ModelsComparison from "./ModelsComparison";

// Import the read-only components

export function PredictionTab() {
  const { predictions, inputData, setPredictions, setInputData } =
    usePredictionContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load predictions from sessionStorage on mount
  useEffect(() => {
    const savedPredictions = sessionStorage.getItem("predictions");
    const savedInputData = sessionStorage.getItem("inputData");

    if (savedPredictions && !predictions) {
      setPredictions(JSON.parse(savedPredictions));
    }
    if (savedInputData && !inputData) {
      setInputData(JSON.parse(savedInputData));
    }
  }, []);

  const handlePredict = async (features: number[]) => {
    setLoading(true);
    setError(null);
    setInputData(features);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      const response = await fetch(`${apiUrl}/api/predict/compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        throw new Error(`Failed to get predictions: ${response.status}`);
      }

      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      setError(errorMessage);
      setPredictions(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClearResults = () => {
    setPredictions(null);
    setInputData(null);
    setError(null);
  };

  return (
    <div className="space-y-8">
      <PredictionForm onPredict={handlePredict} loading={loading} />

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-200">
          <p className="font-semibold">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {predictions && inputData && (
        <>
          <div className="flex justify-end">
            <Button
              onClick={handleClearResults}
              variant="outline"
              className="bg-slate-800/50 border-slate-700/50 text-slate-300 hover:bg-slate-700/50"
            >
              Clear Results
            </Button>
          </div>
          <ResultsDashboard predictions={predictions} />
          <ModelsComparison predictions={predictions} inputData={inputData} />
        </>
      )}
    </div>
  );
}
