"use client";

import { createContext, useContext, useState, type ReactNode } from "react";

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

interface PredictionData {
  modelCount: number;
  predictions: Record<string, PredictionResult>;
}

interface PredictionContextType {
  predictions: PredictionData | null;
  inputData: number[] | null;
  setPredictions: (predictions: PredictionData | null) => void;
  setInputData: (data: number[] | null) => void;
  clearPredictions: () => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(
  undefined,
);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [predictions, setPredictionsState] = useState<PredictionData | null>(
    null,
  );
  const [inputData, setInputDataState] = useState<number[] | null>(null);

  const setPredictions = (data: PredictionData | null) => {
    setPredictionsState(data);
    // Persist to sessionStorage
    if (data) {
      sessionStorage.setItem("predictions", JSON.stringify(data));
    } else {
      sessionStorage.removeItem("predictions");
    }
  };

  const setInputData = (data: number[] | null) => {
    setInputDataState(data);
    // Persist to sessionStorage
    if (data) {
      sessionStorage.setItem("inputData", JSON.stringify(data));
    } else {
      sessionStorage.removeItem("inputData");
    }
  };

  const clearPredictions = () => {
    setPredictionsState(null);
    setInputDataState(null);
    sessionStorage.removeItem("predictions");
    sessionStorage.removeItem("inputData");
  };

  return (
    <PredictionContext.Provider
      value={{
        predictions,
        inputData,
        setPredictions,
        setInputData,
        clearPredictions,
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
}

export function usePredictionContext() {
  const context = useContext(PredictionContext);
  if (context === undefined) {
    throw new Error(
      "usePredictionContext must be used within a PredictionProvider",
    );
  }
  return context;
}
