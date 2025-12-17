"use client";

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

interface ResultsDashboardProps {
  predictions: {
    modelCount: number;
    predictions: Record<string, PredictionResult>;
  };
}

export default function ResultsDashboard({
  predictions,
}: ResultsDashboardProps) {
  const isDeepLearningModel = (modelName: string) => {
    return (
      modelName.toLowerCase().includes("cnn") ||
      modelName.toLowerCase().includes("lstm")
    );
  };

  const modelPredictions = Object.entries(predictions.predictions)
    .filter(([_, data]) => !("error" in data))
    .map(([modelName, data]) => ({
      modelName: modelName.replace(/_/g, " ").toUpperCase(),
      originalName: modelName,
      ...(data as PredictionResult),
    }));

  const riskCount = modelPredictions.filter((m) => m.prediction === 1).length;

  const getRiskLevel = (riskCount: number, total: number) => {
    const percentage = (riskCount / total) * 100;
    if (percentage >= 80)
      return {
        label: "HIGH RISK",
        color: "bg-red-500/20 border-red-500/50 text-red-300",
      };
    if (percentage >= 50)
      return {
        label: "MODERATE RISK",
        color: "bg-yellow-500/20 border-yellow-500/50 text-yellow-300",
      };
    return {
      label: "LOW RISK",
      color: "bg-green-500/20 border-green-500/50 text-green-300",
    };
  };

  const riskLevel = getRiskLevel(riskCount, modelPredictions.length);

  return (
    <div className="space-y-6">
      <div className={`border rounded-xl p-8 ${riskLevel.color}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium opacity-75">
              Consensus Assessment
            </p>
            <h3 className="text-3xl font-bold mt-2">{riskLevel.label}</h3>
            <p className="text-sm mt-2 opacity-75">
              {riskCount} of {modelPredictions.length} models indicate risk
            </p>
          </div>
          <div className="text-5xl font-bold opacity-20">
            {Math.round((riskCount / modelPredictions.length) * 100)}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {modelPredictions.map((model) => (
          <Card
            key={model.modelName}
            className="bg-slate-900/50 border-slate-800/50 p-6"
          >
            <div className="space-y-4">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-slate-100">
                    {model.modelName}
                  </h3>
                  <span
                    className={`inline-block mt-1 px-2 py-0.5 rounded text-xs font-medium ${
                      isDeepLearningModel(model.originalName)
                        ? "bg-purple-500/20 text-purple-300"
                        : "bg-blue-500/20 text-blue-300"
                    }`}
                  >
                    {isDeepLearningModel(model.originalName)
                      ? "Deep Learning"
                      : "Machine Learning"}
                  </span>
                </div>
                <div
                  className={`px-3 py-1 rounded-full text-xs font-bold ${
                    model.prediction === 1
                      ? "bg-red-500/30 text-red-300"
                      : "bg-green-500/30 text-green-300"
                  }`}
                >
                  {model.prediction === 1 ? "RISK" : "NO RISK"}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-slate-400">Confidence</span>
                  <span className="text-sm font-bold text-slate-100">
                    {(model.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800/50 rounded-full h-2">
                  <div
                    className={`h-full rounded-full transition-all ${
                      model.prediction === 1 ? "bg-red-500" : "bg-green-500"
                    }`}
                    style={{ width: `${model.probability * 100}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/50">
                <div>
                  <p className="text-xs text-slate-400">Accuracy</p>
                  <p className="font-semibold text-slate-100">
                    {(model.metrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-400">F1 Score</p>
                  <p className="font-semibold text-slate-100">
                    {model.metrics.f1.toFixed(3)}
                  </p>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
