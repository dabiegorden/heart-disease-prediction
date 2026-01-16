"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ExplainableAIChat } from "@/components/explainable-ai-chat";
import { MetricsExplainer } from "@/components/metrics-explainer";
import { Brain, Lightbulb } from "lucide-react";

interface PredictionData {
  allMetrics?: Record<string, any>;
  metrics?: Record<string, any>;
  prediction?: number;
  probability?: number;
  modelName?: string;
}

export function ExplainableAITab() {
  const [data, setData] = useState<PredictionData>({});
  const [loading, setLoading] = useState(false);

  // Fetch latest prediction data from sessionStorage or API
  useEffect(() => {
    const savedPredictions = sessionStorage.getItem("predictions");
    if (savedPredictions) {
      try {
        const predictions = JSON.parse(savedPredictions);
        setData({
          allMetrics: predictions.predictions,
        });
      } catch (error) {
        console.error("Failed to parse prediction data:", error);
      }
    }
  }, []);

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <div className="flex items-center gap-3 mb-4">
          <Brain className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-xl font-bold text-white">
              Explainable AI Assistant
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Understand model metrics and predictions through interactive AI
            </p>
          </div>
        </div>

        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 text-cyan-200">
          <div className="flex gap-2">
            <Lightbulb className="w-4 h-4 mt-0.5 shrink-0" />
            <p className="text-xs">
              Ask questions about what metrics mean, why models made certain
              predictions, or get explanations about the model's performance
            </p>
          </div>
        </div>
      </Card>

      <Tabs defaultValue="chat" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-slate-900/50 border border-slate-800/50">
          <TabsTrigger
            value="chat"
            className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
          >
            Ask AI Questions
          </TabsTrigger>
          <TabsTrigger
            value="metrics"
            className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
          >
            Learn Metrics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="mt-6">
          <Card className="bg-slate-900/50 border-slate-800/50 p-6">
            <div className="h-96 bg-slate-950/50 rounded-lg border border-slate-800/50 overflow-hidden">
              <ExplainableAIChat
                allMetrics={data.allMetrics}
                metrics={data.metrics}
                prediction={data.prediction}
                probability={data.probability}
                modelName={data.modelName}
              />
            </div>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800/50 p-6 mt-6">
            <h3 className="text-sm font-semibold text-cyan-300 mb-3">
              Example Questions:
            </h3>
            <ul className="space-y-2 text-xs text-slate-400">
              <li>✓ "What does accuracy mean in heart disease prediction?"</li>
              <li>✓ "Why is precision important for medical models?"</li>
              <li>✓ "Which model should I trust more and why?"</li>
              <li>
                ✓ "What's the difference between sensitivity and specificity?"
              </li>
              <li>✓ "How confident is the model in this prediction?"</li>
            </ul>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card className="bg-slate-900/50 border-slate-800/50 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Metric Definitions
                </h3>
                <div className="space-y-4 text-sm text-slate-300">
                  <div className="border-l-2 border-cyan-500/50 pl-4">
                    <p className="font-semibold text-cyan-300">Accuracy</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Percentage of correct predictions out of total predictions
                    </p>
                  </div>
                  <div className="border-l-2 border-blue-500/50 pl-4">
                    <p className="font-semibold text-blue-300">Precision</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Of all positive predictions, how many were actually
                      correct
                    </p>
                  </div>
                  <div className="border-l-2 border-purple-500/50 pl-4">
                    <p className="font-semibold text-purple-300">Recall</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Of all actual positive cases, how many did the model catch
                    </p>
                  </div>
                  <div className="border-l-2 border-green-500/50 pl-4">
                    <p className="font-semibold text-green-300">F1 Score</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Balanced measure combining precision and recall
                    </p>
                  </div>
                  <div className="border-l-2 border-yellow-500/50 pl-4">
                    <p className="font-semibold text-yellow-300">AUC-ROC</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Model's ability to distinguish between positive and
                      negative cases across all thresholds
                    </p>
                  </div>
                </div>
              </Card>
            </div>

            <div>
              <MetricsExplainer
                allMetrics={data.allMetrics}
                metrics={data.metrics}
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
