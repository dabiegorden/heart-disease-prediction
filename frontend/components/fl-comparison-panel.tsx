"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export function FLComparisonPanel() {
  const [comparison, setComparison] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchComparison = async () => {
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
        const response = await fetch(`${apiUrl}/api/federated/comparison`);
        const data = await response.json();

        if (data.success) {
          setComparison(data.comparison);
        }
      } catch (error) {
        console.error("Failed to fetch comparison:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchComparison();
  }, []);

  if (loading) {
    return (
      <Card className="bg-slate-900/50 border-slate-800/50 p-8">
        <div className="text-center text-slate-400">
          <p>Loading comparison data...</p>
        </div>
      </Card>
    );
  }

  if (!comparison) {
    return (
      <Card className="bg-slate-900/50 border-slate-800/50 p-8">
        <div className="text-center text-slate-400">
          <p>No comparison data available</p>
          <p className="text-sm mt-2">
            Train models with FL and run evaluation
          </p>
        </div>
      </Card>
    );
  }

  const chartData = Object.keys(comparison.centralized).map((model) => ({
    model: model.replace(/_/g, " ").toUpperCase(),
    centralized: comparison.centralized[model].accuracy * 100,
    federated: comparison.federated[model]?.accuracy * 100 || 0,
  }));

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Accuracy Comparison: FL vs Centralized
        </h3>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148, 163, 184, 0.1)"
              />
              <XAxis
                dataKey="model"
                tick={{ fill: "rgba(148, 163, 184, 0.7)", fontSize: 12 }}
              />
              <YAxis
                label={{
                  value: "Accuracy (%)",
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
              <Bar
                dataKey="centralized"
                fill="#3b82f6"
                name="Centralized"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="federated"
                fill="#e67e22"
                name="Federated"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card className="bg-slate-900/50 border-slate-800/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">
          Detailed Metrics Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800/50">
              <tr>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">
                  Model
                </th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">
                  Centralized
                </th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">
                  Federated
                </th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">
                  Difference
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(comparison.centralized).map((model) => {
                const centAcc = comparison.centralized[model].accuracy * 100;
                const fedAcc = comparison.federated[model]?.accuracy * 100 || 0;
                const diff = fedAcc - centAcc;

                return (
                  <tr
                    key={model}
                    className="border-b border-slate-800/30 hover:bg-slate-800/20"
                  >
                    <td className="py-3 px-4 font-medium text-slate-200">
                      {model.replace(/_/g, " ").toUpperCase()}
                    </td>
                    <td className="py-3 px-4 text-center text-blue-300">
                      {centAcc.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-center text-orange-300">
                      {fedAcc.toFixed(2)}%
                    </td>
                    <td
                      className={`py-3 px-4 text-center font-semibold ${
                        diff >= 0 ? "text-green-300" : "text-red-300"
                      }`}
                    >
                      {diff >= 0 ? "+" : ""}
                      {diff.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
