"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Card } from "@/components/ui/card";

interface PredictionResult {
  prediction: number;
  probability: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auc: number;
  };
}

interface ModelsComparisonProps {
  predictions: {
    predictions: Record<string, PredictionResult>;
  };
  inputData: number[];
}

export default function ModelsComparison({
  predictions,
  inputData,
}: ModelsComparisonProps) {
  const modelMetrics = Object.entries(predictions.predictions)
    .filter(([_, data]) => !("error" in data))
    .map(([modelName, data]) => ({
      model: modelName.replace(/_/g, " ").toUpperCase(),
      accuracy: (data as PredictionResult).metrics.accuracy * 100,
      precision: (data as PredictionResult).metrics.precision * 100,
      recall: (data as PredictionResult).metrics.recall * 100,
      f1: (data as PredictionResult).metrics.f1 * 100,
      auc: ((data as PredictionResult).metrics.auc || 0) * 100,
    }));

  const metricsForRadar = modelMetrics.map((m) => ({
    model: m.model,
    accuracy: m.accuracy,
    precision: m.precision,
    recall: m.recall,
    auc: m.auc,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-900/50 border-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Model Performance Metrics
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetrics}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148, 163, 184, 0.1)"
                />
                <XAxis
                  dataKey="model"
                  tick={{ fill: "rgba(148, 163, 184, 0.7)", fontSize: 12 }}
                />
                <YAxis
                  tick={{ fill: "rgba(148, 163, 184, 0.7)", fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(15, 23, 42, 0.9)",
                    border: "1px solid rgba(71, 85, 105, 0.5)",
                  }}
                  labelStyle={{ color: "rgba(226, 232, 240, 0.9)" }}
                />
                <Legend />
                <Bar
                  dataKey="accuracy"
                  fill="#3b82f6"
                  name="Accuracy"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="precision"
                  fill="#06b6d4"
                  name="Precision"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="recall"
                  fill="#8b5cf6"
                  name="Recall"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Model Strengths Comparison
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={metricsForRadar}>
                <PolarGrid stroke="rgba(71, 85, 105, 0.3)" />
                <PolarAngleAxis
                  dataKey="model"
                  tick={{ fill: "rgba(148, 163, 184, 0.7)", fontSize: 10 }}
                />
                <PolarRadiusAxis
                  tick={{ fill: "rgba(148, 163, 184, 0.5)", fontSize: 10 }}
                />
                <Radar
                  name="Accuracy"
                  dataKey="accuracy"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.2}
                />
                <Radar
                  name="Precision"
                  dataKey="precision"
                  stroke="#06b6d4"
                  fill="#06b6d4"
                  fillOpacity={0.2}
                />
                <Radar
                  name="AUC"
                  dataKey="auc"
                  stroke="#8b5cf6"
                  fill="#8b5cf6"
                  fillOpacity={0.2}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">
          Detailed Model Metrics
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800/50">
              <tr>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">
                  Model
                </th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">
                  Accuracy
                </th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">
                  Precision
                </th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">
                  Recall
                </th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">
                  F1 Score
                </th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">
                  AUC-ROC
                </th>
              </tr>
            </thead>
            <tbody>
              {modelMetrics.map((model) => (
                <tr
                  key={model.model}
                  className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                >
                  <td className="py-3 px-4 font-medium text-slate-200">
                    {model.model}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {model.accuracy.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {model.precision.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {model.recall.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {(
                      Object.values(predictions.predictions).find(
                        (p) => !("error" in p)
                      )!.metrics.f1 * 100
                    ).toFixed(1)}
                    %
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {model.auc.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
