"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FLTrainingPanel } from "@/components/fl-training-panel";
import { FLResultsPanel } from "@/components/fl-results-panel";
import { FLComparisonPanel } from "@/components/fl-comparison-panel";

export function FederatedLearningTab() {
  const [results, setResults] = useState<any>(null);

  const fetchResults = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const response = await fetch(`${apiUrl}/api/retrain/results`);
      const data = await response.json();
      if (data.success) {
        setResults(data);
      }
    } catch (error) {
      console.error("Failed to fetch training results:", error);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white">
              Model Retraining System
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Upload your own dataset to retrain models with custom data
            </p>
          </div>
          <div className="flex gap-2">
            {results?.completed > 0 && (
              <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
                {results.completed} Trained
              </Badge>
            )}
            {results?.total > results?.completed && (
              <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 animate-pulse">
                {results.total - results.completed} In Progress
              </Badge>
            )}
          </div>
        </div>

        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 text-cyan-200">
          <p className="font-semibold text-sm">How It Works</p>
          <p className="text-xs mt-1">
            Upload a CSV dataset with your custom data. The system will train
            the selected model(s) and provide performance metrics. You can train
            individual models or all 6 models at once.
          </p>
        </div>
      </Card>

      <Tabs defaultValue="training" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-900/50 border border-slate-800/50">
          <TabsTrigger
            value="training"
            className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
          >
            Training
          </TabsTrigger>
          <TabsTrigger
            value="results"
            className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
          >
            Results
          </TabsTrigger>
          <TabsTrigger
            value="comparison"
            className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
          >
            Model Comparison
          </TabsTrigger>
        </TabsList>

        <TabsContent value="training" className="mt-6">
          <FLTrainingPanel status={results} onRefresh={fetchResults} />
        </TabsContent>

        <TabsContent value="results" className="mt-6">
          <FLResultsPanel />
        </TabsContent>

        <TabsContent value="comparison" className="mt-6">
          <FLComparisonPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
