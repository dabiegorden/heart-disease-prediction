"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

interface MetricsExplainerProps {
  metrics?: Record<string, any>;
  allMetrics?: Record<string, any>;
}

const METRIC_NAMES = [
  "Accuracy",
  "Precision",
  "Recall",
  "F1-Score",
  "AUC-ROC",
  "Specificity",
  "Sensitivity",
];

export function MetricsExplainer({
  metrics,
  allMetrics,
}: MetricsExplainerProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleExplainMetric = async (metricName: string) => {
    setSelectedMetric(metricName);
    setLoading(true);
    setExplanation("");

    try {
      const response = await fetch(
        `http://localhost:5000/api/explainable-ai/explain-metrics`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            metricName,
            metrics,
            allMetrics,
          }),
        }
      );

      const data = await response.json();

      if (data.success) {
        setExplanation(data.explanation);
      } else {
        setExplanation(`Error: ${data.error}`);
      }
    } catch (error) {
      setExplanation("Failed to load explanation. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 p-6">
      <div className="flex items-center gap-2 mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">
            Learn About Metrics
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Click on any metric to get an AI explanation
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-6">
        {METRIC_NAMES.map((metric) => (
          <Button
            key={metric}
            variant={selectedMetric === metric ? "default" : "outline"}
            onClick={() => handleExplainMetric(metric)}
            disabled={loading}
            className={`text-xs ${
              selectedMetric === metric
                ? "bg-cyan-600/80 hover:bg-cyan-700 text-white"
                : "bg-slate-800/50 border-slate-700/50 text-slate-300 hover:bg-slate-700/50"
            }`}
          >
            {metric}
          </Button>
        ))}
      </div>

      {selectedMetric && (
        <div className="mt-6 p-4 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
          <h4 className="font-semibold text-cyan-300 mb-3">{selectedMetric}</h4>
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span className="text-sm text-slate-400">
                Loading explanation...
              </span>
            </div>
          ) : (
            <p className="text-sm text-slate-300 whitespace-pre-wrap">
              {explanation}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
