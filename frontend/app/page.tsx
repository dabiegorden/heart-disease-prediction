"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PredictionProvider } from "@/components/prediction-context";
import { PredictionTab } from "@/components/prediction-tab";
import { FederatedLearningTab } from "@/components/federated-learning-tab";
import { ExplainableAITab } from "@/components/explainable-ai-tab";

function Header() {
  return (
    <header className="border-b border-slate-800/50 bg-slate-950/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-linear-to-r from-blue-400 via-cyan-400 to-blue-500">
            CardioPredict AI
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Federated Learning Clinical Heart Disease Risk Assessment
          </p>
        </div>
        <div className="text-right text-sm text-slate-400">
          <p>Powered by 7 AI Models</p>
          <p className="text-xs mt-1">5 ML + 2 Deep Learning</p>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("prediction");

  return (
    <PredictionProvider>
      <div className="min-h-screen bg-linear-to-br from-slate-950 via-slate-900 to-slate-950">
        <Header />
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 bg-slate-900/50 border border-slate-800/50 mb-8">
              <TabsTrigger
                value="prediction"
                className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
              >
                Prediction
              </TabsTrigger>
              <TabsTrigger
                value="federated"
                className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
              >
                Model Training
              </TabsTrigger>
              <TabsTrigger
                value="explainable-ai"
                className="data-[state=active]:bg-cyan-500/20 text-white data-[state=active]:text-cyan-300"
              >
                Explainable AI
              </TabsTrigger>
            </TabsList>

            <TabsContent value="prediction">
              <PredictionTab />
            </TabsContent>

            <TabsContent value="federated">
              <FederatedLearningTab />
            </TabsContent>

            <TabsContent value="explainable-ai">
              <ExplainableAITab />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </PredictionProvider>
  );
}
