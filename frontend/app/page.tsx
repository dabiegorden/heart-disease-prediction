"use client";

import {
  Header,
  ModelsComparison,
  PredictionForm,
  ResultsDashboard,
} from "@/constants";
import { useState } from "react";

export default function Home() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputData, setInputData] = useState<number[] | null>(null);

  const handlePredict = async (features: number[]) => {
    setLoading(true);
    setError(null);
    setInputData(features);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      if (!apiUrl) {
        throw new Error(
          "API URL not configured. Set NEXT_PUBLIC_API_URL environment variable."
        );
      }

      console.log(
        "Sending prediction request to:",
        `${apiUrl}/api/predict/compare`
      );
      console.log("Features:", features);

      const response = await fetch(`${apiUrl}/api/predict/compare`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ features }),
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Response error:", errorText);
        throw new Error(`Failed to get predictions: ${response.status}`);
      }

      const data = await response.json();
      console.log("Prediction data received:", data);
      setPredictions(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      console.error("Prediction error:", errorMessage);
      setError(errorMessage);
      setPredictions(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-slate-950 via-slate-900 to-slate-950">
      <Header />

      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        <PredictionForm onPredict={handlePredict} loading={loading} />

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-200">
            <p className="font-semibold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {predictions && inputData && (
          <>
            <ResultsDashboard predictions={predictions} />
            <ModelsComparison predictions={predictions} inputData={inputData} />
          </>
        )}
      </div>
    </main>
  );
}
