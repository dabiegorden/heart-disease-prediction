"use client";

import type React from "react";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Upload, Loader2 } from "lucide-react";

const MODELS = [
  { id: "logistic_regression", name: "Logistic Regression", type: "ML" },
  { id: "svm", name: "Support Vector Machine", type: "ML" },
  { id: "gradient_boost", name: "Gradient Boosting", type: "ML" },
  { id: "knn", name: "K-Nearest Neighbors", type: "ML" },
  { id: "cnn1d", name: "1D CNN", type: "DL" },
  { id: "cnn_lstm", name: "CNN + LSTM", type: "DL" },
];

export function FLTrainingPanel({ status, onRefresh }: any) {
  const [selectedModel, setSelectedModel] = useState("logistic_regression");
  const [epochs, setEpochs] = useState(50);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: string; text: string } | null>(
    null
  );
  const [trainingProgress, setTrainingProgress] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        setDatasetFile(file);
        setMessage({ type: "success", text: `Dataset loaded: ${file.name}` });
      } else {
        setMessage({ type: "error", text: "Please upload a CSV file" });
      }
    }
  };

  const handleTrainModel = async () => {
    if (!datasetFile) {
      setMessage({ type: "error", text: "Please upload a dataset file first" });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const formData = new FormData();
      formData.append("dataset", datasetFile);
      formData.append("modelType", selectedModel);
      formData.append("epochs", epochs.toString());

      console.log(`[v0] Training ${selectedModel} with uploaded dataset`);

      const response = await fetch(`${apiUrl}/api/retrain/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setMessage({
          type: "success",
          text: `Training started for ${selectedModel}. Session ID: ${data.sessionId}`,
        });

        // Poll for training status
        pollTrainingStatus(data.sessionId);
      } else {
        setMessage({ type: "error", text: data.error || "Training failed" });
      }
    } catch (error) {
      console.error("Training error:", error);
      setMessage({ type: "error", text: "Failed to start training" });
    } finally {
      setLoading(false);
    }
  };

  const handleTrainAll = async () => {
    if (!datasetFile) {
      setMessage({ type: "error", text: "Please upload a dataset file first" });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const formData = new FormData();
      formData.append("dataset", datasetFile);
      formData.append("epochs", epochs.toString());

      console.log("[v0] Training all 6 models with uploaded dataset");

      const response = await fetch(`${apiUrl}/api/retrain/train-all`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setMessage({
          type: "success",
          text: `Training all ${data.totalModels} models started! Session ID: ${data.sessionId}`,
        });

        // Poll for training status
        pollTrainingStatus(data.sessionId);
      } else {
        setMessage({ type: "error", text: data.error || "Training failed" });
      }
    } catch (error) {
      console.error("Training error:", error);
      setMessage({ type: "error", text: "Failed to start training" });
    } finally {
      setLoading(false);
    }
  };

  const pollTrainingStatus = async (sessionId: string) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
    const maxAttempts = 60;
    let attempts = 0;

    const poll = setInterval(async () => {
      try {
        const response = await fetch(
          `${apiUrl}/api/retrain/status/${sessionId}`
        );
        const data = await response.json();

        if (data.success) {
          setTrainingProgress(data.session);

          if (data.session.status === "completed") {
            clearInterval(poll);
            setMessage({
              type: "success",
              text: "Training completed successfully!",
            });
            onRefresh();
          } else if (data.session.status === "failed") {
            clearInterval(poll);
            setMessage({
              type: "error",
              text: `Training failed: ${data.session.error}`,
            });
          }
        }

        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(poll);
        }
      } catch (error) {
        console.error("Status poll error:", error);
      }
    }, 3000);
  };

  return (
    <div className="space-y-6">
      {message && (
        <div
          className={`rounded-lg p-4 ${
            message.type === "success"
              ? "bg-green-500/10 border border-green-500/30 text-green-200"
              : "bg-red-500/10 border border-red-500/30 text-red-200"
          }`}
        >
          <p className="text-sm">{message.text}</p>
        </div>
      )}

      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Upload Training Dataset
        </h3>
        <div className="space-y-4">
          <div>
            <Label className="text-slate-200">Dataset (CSV format)</Label>
            <div className="mt-2">
              <label
                htmlFor="dataset-upload"
                className="flex items-center justify-center w-full p-6 border-2 border-dashed border-slate-700 rounded-lg cursor-pointer hover:border-cyan-500 transition-colors bg-slate-800/30"
              >
                <div className="text-center">
                  <Upload className="mx-auto h-12 w-12 text-slate-400 mb-2" />
                  {datasetFile ? (
                    <div>
                      <p className="text-sm font-semibold text-cyan-300">
                        {datasetFile.name}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        {(datasetFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-sm text-slate-300">
                        Click to upload dataset
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        CSV file (max 50MB)
                      </p>
                    </div>
                  )}
                </div>
                <input
                  id="dataset-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>
          </div>
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <p className="text-xs text-blue-200">
              <strong>Dataset Format:</strong> CSV file with features as columns
              and the target variable (0 or 1) as the last column. The system
              will automatically split and normalize the data.
            </p>
          </div>
        </div>
      </Card>

      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Model Training Configuration
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <Label className="text-slate-200">Select Model</Label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full mt-2 bg-slate-800/50 border border-slate-700/50 text-white rounded-lg p-2"
              disabled={!datasetFile || loading}
            >
              {MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.type})
                </option>
              ))}
            </select>
          </div>
          <div>
            <Label className="text-slate-200">
              Training Epochs (DL models only)
            </Label>
            <Input
              type="number"
              min={10}
              max={200}
              value={epochs}
              onChange={(e) => setEpochs(Number.parseInt(e.target.value))}
              className="bg-slate-800/50 border-slate-700/50 text-white mt-2"
              disabled={!datasetFile || loading}
            />
          </div>
        </div>
        <div className="flex gap-4">
          <Button
            onClick={handleTrainModel}
            disabled={loading || !datasetFile}
            className="flex-1 bg-linear-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Training...
              </>
            ) : (
              "Train Selected Model"
            )}
          </Button>
          <Button
            onClick={handleTrainAll}
            disabled={loading || !datasetFile}
            className="flex-1 bg-linear-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Training All...
              </>
            ) : (
              "Train All 6 Models"
            )}
          </Button>
        </div>
      </Card>

      {trainingProgress && (
        <Card className="bg-slate-900/50 border-slate-800/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Training Progress
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-200">Status:</span>
              <Badge
                className={
                  trainingProgress.status === "completed"
                    ? "bg-green-500/20 text-green-300 border-green-500/30"
                    : trainingProgress.status === "failed"
                    ? "bg-red-500/20 text-red-300 border-red-500/30"
                    : "bg-blue-500/20 text-blue-300 border-blue-500/30 animate-pulse"
                }
              >
                {trainingProgress.status}
              </Badge>
            </div>
            {trainingProgress.progress !== undefined && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-slate-300">
                  <span>Progress</span>
                  <span>{trainingProgress.progress}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                    className="bg-linear-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${trainingProgress.progress}%` }}
                  />
                </div>
              </div>
            )}
            {trainingProgress.metrics && (
              <div className="mt-4 grid grid-cols-2 gap-3">
                {Object.entries(trainingProgress.metrics).map(
                  ([key, value]: [string, any]) => (
                    <div key={key} className="bg-slate-800/30 rounded p-3">
                      <p className="text-xs text-slate-400 capitalize">
                        {key.replace("_", " ")}
                      </p>
                      <p className="text-lg font-semibold text-white">
                        {(value * 100).toFixed(2)}%
                      </p>
                    </div>
                  )
                )}
              </div>
            )}
            {trainingProgress.results && (
              <div className="mt-4">
                <p className="text-sm text-slate-300 font-semibold mb-2">
                  All Models Results:
                </p>
                <div className="space-y-2">
                  {Object.entries(trainingProgress.results).map(
                    ([model, result]: [string, any]) => (
                      <div key={model} className="bg-slate-800/30 rounded p-3">
                        <p className="text-xs text-slate-400 mb-1">{model}</p>
                        {result.error ? (
                          <p className="text-xs text-red-300">{result.error}</p>
                        ) : (
                          <p className="text-sm text-white">
                            Accuracy: {(result.accuracy * 100).toFixed(2)}% |
                            F1: {(result.f1_score * 100).toFixed(2)}%
                          </p>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
