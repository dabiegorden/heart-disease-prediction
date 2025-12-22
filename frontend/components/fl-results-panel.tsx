"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

export function FLResultsPanel() {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

      console.log("[v0] Fetching training results");

      const response = await fetch(`${apiUrl}/api/retrain/results`);
      const data = await response.json();

      console.log("[v0] Results data:", data);

      if (data.success && data.sessions.length > 0) {
        const latestSession = data.sessions[data.sessions.length - 1];
        console.log("[v0] Latest session:", latestSession);
        setResults(latestSession);
      } else {
        console.log("[v0] No sessions found");
        setResults(null);
      }
    } catch (error) {
      console.error("[v0] Failed to fetch results:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchResults, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !results) {
    return (
      <Card className="bg-slate-900/50 border-slate-800/50 p-8">
        <div className="text-center text-slate-400">
          <p>Loading results...</p>
        </div>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card className="bg-slate-900/50 border-slate-800/50 p-8">
        <div className="text-center text-slate-400">
          <p>No training results yet</p>
          <p className="text-sm mt-2">Train models to see results</p>
        </div>
      </Card>
    );
  }

  const modelNames =
    results.modelType === "all"
      ? Object.keys(results.results || {})
      : [results.modelType];

  const chartData = modelNames.map((modelName) => {
    const modelStr = String(modelName);
    const metrics =
      results.modelType === "all"
        ? results.results[modelName]
        : results.metrics;

    // Skip if metrics are missing or contain errors
    if (!metrics || metrics.error) {
      return {
        name: modelStr.replace(/_/g, " ").toUpperCase(),
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1_score: 0,
      };
    }

    return {
      name: modelStr.replace(/_/g, " ").toUpperCase(),
      accuracy: (metrics.accuracy || 0) * 100,
      precision: (metrics.precision || 0) * 100,
      recall: (metrics.recall || 0) * 100,
      f1_score: (metrics.f1_score || 0) * 100,
    };
  });

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Training Status
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/30 rounded-lg p-4">
            <p className="text-xs text-slate-400">Status</p>
            <p className="text-2xl font-bold text-cyan-400 mt-1 capitalize">
              {results.status}
            </p>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <p className="text-xs text-slate-400">Progress</p>
            <p className="text-2xl font-bold text-blue-400 mt-1">
              {results.progress}%
            </p>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <p className="text-xs text-slate-400">Models Trained</p>
            <p className="text-2xl font-bold text-purple-400 mt-1">
              {modelNames.length}
            </p>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <p className="text-xs text-slate-400">Session ID</p>
            <p className="text-sm font-bold text-green-400 mt-1">
              {results.sessionId}
            </p>
          </div>
        </div>
      </Card>

      {results.status === "completed" && (
        <Card className="bg-slate-900/50 border-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Model Performance Comparison
          </h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148, 163, 184, 0.1)"
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "rgba(148, 163, 184, 0.7)", fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis
                  label={{
                    value: "Score (%)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                  tick={{ fill: "rgba(148, 163, 184, 0.7)" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(15, 23, 42, 0.9)",
                    border: "1px solid rgba(71, 85, 105, 0.5)",
                  }}
                />
                <Legend />
                <Bar dataKey="accuracy" fill="#06b6d4" name="Accuracy" />
                <Bar dataKey="precision" fill="#8b5cf6" name="Precision" />
                <Bar dataKey="recall" fill="#10b981" name="Recall" />
                <Bar dataKey="f1_score" fill="#f59e0b" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {results.status === "completed" && (
        <Card className="bg-slate-900/50 border-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Detailed Metrics
          </h3>
          <div className="space-y-4">
            {modelNames.map((modelName) => {
              const modelStr = String(modelName);
              const metrics =
                results.modelType === "all"
                  ? results.results[modelName]
                  : results.metrics;

              if (!metrics || metrics.error) {
                return (
                  <div
                    key={modelStr}
                    className="bg-slate-800/30 rounded-lg p-4"
                  >
                    <h4 className="font-semibold text-red-400 mb-2">
                      {modelStr.replace(/_/g, " ").toUpperCase()}
                    </h4>
                    <p className="text-sm text-slate-400">
                      Error: {metrics?.error || "Training failed"}
                    </p>
                  </div>
                );
              }

              return (
                <div key={modelStr} className="bg-slate-800/30 rounded-lg p-4">
                  <h4 className="font-semibold text-cyan-400 mb-3">
                    {modelStr.replace(/_/g, " ").toUpperCase()}
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <div>
                      <p className="text-xs text-slate-400">Accuracy</p>
                      <p className="text-lg font-bold text-white">
                        {((metrics.accuracy || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Precision</p>
                      <p className="text-lg font-bold text-white">
                        {((metrics.precision || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Recall</p>
                      <p className="text-lg font-bold text-white">
                        {((metrics.recall || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">F1 Score</p>
                      <p className="text-lg font-bold text-white">
                        {((metrics.f1_score || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">AUC-ROC</p>
                      <p className="text-lg font-bold text-white">
                        {((metrics.auc_roc || 0) * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {results.error && (
        <Card className="bg-red-900/20 border-red-800/50 p-6">
          <h3 className="text-lg font-semibold text-red-400 mb-2">
            Training Error
          </h3>
          <p className="text-sm text-slate-300">{results.error}</p>
        </Card>
      )}
    </div>
  );
}
