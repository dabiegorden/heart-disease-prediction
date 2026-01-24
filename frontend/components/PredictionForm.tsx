"use client";

import type React from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const FEATURES = [
  { id: "age", label: "Age", placeholder: "30-80", min: 18, max: 100 },
  {
    id: "sex",
    label: "Sex (0=F, 1=M)",
    placeholder: "0 or 1",
    min: 0,
    max: 1,
    step: 1,
  },
  {
    id: "chestpaintype",
    label: "Chest Pain Type (0-3)",
    placeholder: "0-3",
    min: 0,
    max: 3,
    step: 1,
  },
  {
    id: "restingbps",
    label: "Resting BP (mmHg)",
    placeholder: "90-180",
    min: 50,
    max: 200,
  },
  {
    id: "cholesterol",
    label: "Cholesterol (mg/dL)",
    placeholder: "100-400",
    min: 0,
    max: 500,
  },
  {
    id: "fastingbloodsugar",
    label: "Fasting Blood Sugar (0=no, 1=yes)",
    placeholder: "0 or 1",
    min: 0,
    max: 1,
    step: 1,
  },
  {
    id: "restingecg",
    label: "Resting ECG (0-2)",
    placeholder: "0-2",
    min: 0,
    max: 2,
    step: 1,
  },
  {
    id: "maxheartrate",
    label: "Max Heart Rate (bpm)",
    placeholder: "60-200",
    min: 40,
    max: 220,
  },
  {
    id: "exerciseangina",
    label: "Exercise Angina (0=no, 1=yes)",
    placeholder: "0 or 1",
    min: 0,
    max: 1,
    step: 1,
  },
  {
    id: "oldpeak",
    label: "ST Depression (Oldpeak)",
    placeholder: "0-6",
    min: 0,
    max: 10,
    step: 0.1,
  },
  {
    id: "slope",
    label: "ST Slope (0-2)",
    placeholder: "0-2",
    min: 0,
    max: 2,
    step: 1,
  },
  {
    id: "noofmajorvessels",
    label: "Major Vessels (0-4)",
    placeholder: "0-4",
    min: 0,
    max: 4,
    step: 1,
  },
];

export default function PredictionForm({
  onPredict,
  loading,
}: {
  onPredict: (features: number[]) => void;
  loading: boolean;
}) {
  const [values, setValues] = useState<Record<string, number | string>>(
    FEATURES.reduce((acc, f) => ({ ...acc, [f.id]: "" }), {}),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (id: string, value: string) => {
    setValues((prev) => ({
      ...prev,
      [id]: value === "" ? "" : Number.parseFloat(value),
    }));
    setErrors((prev) => ({ ...prev, [id]: "" }));
  };

  const validate = () => {
    const newErrors: Record<string, string> = {};
    FEATURES.forEach((feature) => {
      const val = values[feature.id];
      if (val === "") {
        newErrors[feature.id] = "Required";
      } else if (typeof val === "number") {
        if (
          val < (feature.min ?? Number.NEGATIVE_INFINITY) ||
          val > (feature.max ?? Number.POSITIVE_INFINITY)
        ) {
          newErrors[feature.id] = `${feature.min}-${feature.max}`;
        }
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      const features = FEATURES.map((f) => values[f.id] as number);
      onPredict(features);
    }
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-8 backdrop-blur-sm">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">
          Patient Assessment
        </h2>
        <p className="text-slate-400 text-sm">
          Enter clinical parameters for cardiovascular risk evaluation
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((feature) => (
            <div key={feature.id} className="space-y-2">
              <label className="block text-sm font-medium text-slate-200">
                {feature.label}
              </label>
              <div>
                <Input
                  type="number"
                  placeholder={feature.placeholder}
                  value={values[feature.id] === "" ? "" : values[feature.id]}
                  onChange={(e) => handleChange(feature.id, e.target.value)}
                  min={feature.min}
                  max={feature.max}
                  step={feature.step || 1}
                  disabled={loading}
                  className="bg-slate-800/50 border-slate-700/50 text-white placeholder:text-slate-500 focus:border-cyan-500/50 focus:ring-cyan-500/20"
                />
              </div>
              {errors[feature.id] && (
                <p className="text-xs text-red-400">{errors[feature.id]}</p>
              )}
            </div>
          ))}
        </div>

        <Button
          type="submit"
          disabled={loading}
          className="w-full bg-linear-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold h-12 rounded-lg transition-all"
        >
          {loading ? "Analyzing..." : "Run Prediction"}
        </Button>
      </form>
    </div>
  );
}
