/**
 * Explainable AI Routes using Google Gemini 2.5-Flash
 * Provides detailed explanations of metrics, predictions, and model performance
 */

import dotenv from "dotenv";
dotenv.config({ quiet: true });

import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { asyncHandler } from "../middleware/errorHandler.js";

const router = express.Router();

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

/**
 * Format metrics into a comprehensive context for Gemini
 */
function formatMetricsContext(metrics, allMetrics) {
  let context = "## Heart Disease Prediction Models Performance Metrics\n\n";

  if (allMetrics && Object.keys(allMetrics).length > 0) {
    context += "### All Models Overview:\n";
    for (const [modelName, modelMetrics] of Object.entries(allMetrics)) {
      context += `\n**${modelName}:**\n`;
      context += `- Accuracy: ${(modelMetrics.accuracy * 100).toFixed(2)}%\n`;
      context += `- Precision: ${(modelMetrics.precision * 100).toFixed(2)}%\n`;
      context += `- Recall: ${(modelMetrics.recall * 100).toFixed(2)}%\n`;
      context += `- F1-Score: ${modelMetrics.f1_score?.toFixed(4) || "N/A"}\n`;
      context += `- AUC-ROC: ${modelMetrics.auc_roc?.toFixed(4) || "N/A"}\n`;
    }
  }

  if (metrics) {
    context += `\n### Specific Model Metrics:\n`;
    context += `- Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%\n`;
    context += `- Precision: ${(metrics.precision * 100).toFixed(2)}%\n`;
    context += `- Recall: ${(metrics.recall * 100).toFixed(2)}%\n`;
    context += `- F1-Score: ${metrics.f1_score?.toFixed(4) || "N/A"}\n`;
    context += `- AUC-ROC: ${metrics.auc_roc?.toFixed(4) || "N/A"}\n`;
    context += `- Specificity: ${metrics.specificity?.toFixed(4) || "N/A"}\n`;
    context += `- Sensitivity: ${metrics.sensitivity?.toFixed(4) || "N/A"}\n`;
  }

  return context;
}

/**
 * POST /api/explainable-ai/ask
 * Ask Gemini questions about metrics, predictions, and model performance
 */
router.post(
  "/ask",
  asyncHandler(async (req, res) => {
    const {
      question,
      metrics,
      allMetrics,
      prediction,
      probability,
      modelName,
    } = req.body;

    if (!question) {
      return res.status(400).json({
        success: false,
        error: "Question is required",
      });
    }

    try {
      // Build comprehensive context
      let contextPrompt = `You are an expert machine learning doctor/cardiologist explaining heart disease prediction models to medical students and patients. 
Be friendly, clear, and use simple language while maintaining medical accuracy.

${formatMetricsContext(metrics, allMetrics)}`;

      if (prediction !== undefined && probability !== undefined) {
        contextPrompt += `\n### Recent Prediction:\n`;
        contextPrompt += `- Model Used: ${modelName || "Not specified"}\n`;
        contextPrompt += `- Prediction: ${
          prediction === 1 ? "Heart Disease Likely" : "Heart Disease Unlikely"
        }\n`;
        contextPrompt += `- Confidence: ${(probability * 100).toFixed(2)}%\n`;
      }

      contextPrompt += `\n### Context Rules:
1. Explain metrics in plain language (e.g., "Accuracy tells us how often the model is correct")
2. For predictions, explain what the confidence score means
3. Provide medical insights when relevant
4. Be encouraging but honest about model limitations
5. If asked about models you don't have data for, say so clearly`;

      // Stream response if client supports it
      const response = await model.generateContent([
        {
          text: contextPrompt,
        },
        {
          text: `User Question: ${question}\n\nPlease provide a detailed, educational response.`,
        },
      ]);

      const answer = response.response.text();

      res.json({
        success: true,
        question,
        answer,
        modelUsed: "gemini-2.5-flash",
      });
    } catch (error) {
      console.error("Gemini API Error:", error);
      res.status(500).json({
        success: false,
        error: "Failed to get AI explanation",
        details: error.message,
      });
    }
  }),
);

/**
 * POST /api/explainable-ai/explain-metrics
 * Get explanations for specific metrics
 */
router.post(
  "/explain-metrics",
  asyncHandler(async (req, res) => {
    const { metricName, metrics, allMetrics } = req.body;

    if (!metricName) {
      return res.status(400).json({
        success: false,
        error: "Metric name is required",
      });
    }

    try {
      const context = formatMetricsContext(metrics, allMetrics);

      const response = await model.generateContent([
        {
          text: `You are an ML expert explaining evaluation metrics. Keep it concise but thorough.\n\n${context}`,
        },
        {
          text: `Explain the "${metricName}" metric in detail. Include:
1. What it measures
2. Why it's important for heart disease prediction
3. How to interpret the value
4. Ideal range for this use case`,
        },
      ]);

      const explanation = response.response.text();

      res.json({
        success: true,
        metric: metricName,
        explanation,
      });
    } catch (error) {
      console.error("Gemini API Error:", error);
      res.status(500).json({
        success: false,
        error: "Failed to explain metric",
        details: error.message,
      });
    }
  }),
);

/**
 * POST /api/explainable-ai/compare-models
 * Get AI comparison of different models based on metrics
 */
router.post(
  "/compare-models",
  asyncHandler(async (req, res) => {
    const { allMetrics, modelNames } = req.body;

    if (!allMetrics || Object.keys(allMetrics).length === 0) {
      return res.status(400).json({
        success: false,
        error: "Metrics data is required",
      });
    }

    try {
      const context = formatMetricsContext(null, allMetrics);

      const prompt = modelNames
        ? `Compare these models: ${modelNames.join(", ")}`
        : "Compare all available models";

      const response = await model.generateContent([
        {
          text: `You are a medical AI expert. ${context}\n\nProvide a detailed comparison focusing on clinical relevance for heart disease prediction.`,
        },
        {
          text: `${prompt}. Include:
1. Which model is most reliable and why
2. Trade-offs between different models
3. Recommendations for clinical use
4. Any limitations to consider`,
        },
      ]);

      const comparison = response.response.text();

      res.json({
        success: true,
        comparison,
      });
    } catch (error) {
      console.error("Gemini API Error:", error);
      res.status(500).json({
        success: false,
        error: "Failed to compare models",
        details: error.message,
      });
    }
  }),
);

export default router;
