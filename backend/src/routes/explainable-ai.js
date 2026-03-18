/**
 * routes/explainable-ai.js
 * ========================
 * Explainable AI endpoints powered by Google Gemini 2.5-Flash.
 *
 * Every endpoint fetches LIVE data from MongoDB before calling Gemini so
 * the AI always explains the most up-to-date metrics, not stale JSON.
 *
 * Data sources (in priority order)
 * ----------------------------------
 * 1. ModelMetrics collection  – upserted after every training / retrain run
 * 2. Predictions collection   – used to compute live prediction statistics
 * 3. Request body overrides   – callers may still pass extra context
 */

import dotenv from "dotenv";
dotenv.config({ quiet: true });

import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { asyncHandler } from "../middleware/errorHandler.js";
import { ModelMetrics, Prediction, TrainingSession } from "../db/schema.js";

const router = express.Router();

if (!process.env.GEMINI_API_KEY) {
  console.warn(
    "⚠  GEMINI_API_KEY is not set – Explainable AI routes will fail.",
  );
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

/* ============================================================
 * DB HELPERS
 * ============================================================ */

/**
 * Load all model metrics from MongoDB.
 * Returns an object keyed by modelName.
 */
async function fetchAllMetricsFromDB() {
  const records = await ModelMetrics.find({}).sort({ trainedAt: -1 }).lean();

  // Keep only the most-recent record per model (collection may have both
  // "initial" and "retrained" entries; we want the latest one).
  const seen = new Set();
  const metrics = {};

  for (const rec of records) {
    if (!seen.has(rec.modelName)) {
      seen.add(rec.modelName);
      metrics[rec.modelName] = {
        accuracy: rec.accuracy,
        precision: rec.precision,
        recall: rec.recall,
        f1_score: rec.f1 ?? rec.f1_score,
        auc_roc: rec.auc ?? rec.auc_roc,
        cv_mean: rec.cv_mean,
        cv_std: rec.cv_std,
        confusion_matrix: rec.confusion_matrix,
        source: rec.source,
        trainedAt: rec.trainedAt,
      };
    }
  }

  return metrics;
}

/**
 * Load one model's metrics from MongoDB.
 */
async function fetchModelMetricsFromDB(modelName) {
  const rec = await ModelMetrics.findOne({ modelName })
    .sort({ trainedAt: -1 })
    .lean();
  if (!rec) return null;

  return {
    accuracy: rec.accuracy,
    precision: rec.precision,
    recall: rec.recall,
    f1_score: rec.f1 ?? rec.f1_score,
    auc_roc: rec.auc ?? rec.auc_roc,
    cv_mean: rec.cv_mean,
    cv_std: rec.cv_std,
    confusion_matrix: rec.confusion_matrix,
    source: rec.source,
    trainedAt: rec.trainedAt,
  };
}

/**
 * Compute live prediction statistics from the Predictions collection.
 * Returns counts, positive rate, and per-model breakdown.
 */
async function fetchPredictionStats() {
  const [total, positive, perModel] = await Promise.all([
    Prediction.countDocuments({}),
    Prediction.countDocuments({ prediction: 1 }),
    Prediction.aggregate([
      { $match: { modelName: { $ne: "compare" } } },
      {
        $group: {
          _id: "$modelName",
          total: { $sum: 1 },
          positive: { $sum: "$prediction" },
          avgProbability: { $avg: "$probability" },
        },
      },
      { $sort: { total: -1 } },
    ]),
  ]);

  return {
    totalPredictions: total,
    positivePredictions: positive,
    negativePredictions: total - positive,
    positiveRate: total > 0 ? ((positive / total) * 100).toFixed(1) : "N/A",
    perModel: Object.fromEntries(
      perModel.map((m) => [
        m._id,
        {
          total: m.total,
          positive: m.positive,
          avgProbability: (m.avgProbability * 100).toFixed(1) + "%",
        },
      ]),
    ),
  };
}

/* ============================================================
 * CONTEXT BUILDER
 * Build a rich Markdown context block for Gemini from DB data.
 * ============================================================ */
function buildMetricsContext(
  allMetrics,
  predictionStats,
  overrideMetrics = null,
) {
  let ctx = "## Heart Disease Prediction System – Live Data\n\n";

  // ── Model metrics ──────────────────────────────────────────
  if (allMetrics && Object.keys(allMetrics).length > 0) {
    ctx += "### Model Performance Metrics (from database):\n";

    for (const [name, m] of Object.entries(allMetrics)) {
      const trained = m.trainedAt
        ? new Date(m.trainedAt).toLocaleDateString()
        : "unknown";

      ctx += `\n**${name}** (source: ${m.source ?? "initial"}, trained: ${trained})\n`;
      ctx += `- Accuracy  : ${m.accuracy != null ? (m.accuracy * 100).toFixed(2) + "%" : "N/A"}\n`;
      ctx += `- Precision : ${m.precision != null ? (m.precision * 100).toFixed(2) + "%" : "N/A"}\n`;
      ctx += `- Recall    : ${m.recall != null ? (m.recall * 100).toFixed(2) + "%" : "N/A"}\n`;
      ctx += `- F1-Score  : ${m.f1_score != null ? m.f1_score.toFixed(4) : "N/A"}\n`;
      ctx += `- AUC-ROC   : ${m.auc_roc != null ? m.auc_roc.toFixed(4) : "N/A"}\n`;

      if (m.cv_mean != null) {
        ctx += `- CV Mean   : ${(m.cv_mean * 100).toFixed(2)}% (±${(m.cv_std * 100).toFixed(2)}%)\n`;
      }
    }
  } else {
    ctx += "### Model Metrics: No metrics available in the database yet.\n";
  }

  // ── Single model override (passed explicitly by caller) ───
  if (overrideMetrics) {
    ctx += "\n### Highlighted Model Metrics:\n";
    ctx += `- Accuracy    : ${(overrideMetrics.accuracy * 100).toFixed(2)}%\n`;
    ctx += `- Precision   : ${(overrideMetrics.precision * 100).toFixed(2)}%\n`;
    ctx += `- Recall      : ${(overrideMetrics.recall * 100).toFixed(2)}%\n`;
    ctx += `- F1-Score    : ${overrideMetrics.f1_score?.toFixed(4) ?? "N/A"}\n`;
    ctx += `- AUC-ROC     : ${overrideMetrics.auc_roc?.toFixed(4) ?? "N/A"}\n`;
    ctx += `- Specificity : ${overrideMetrics.specificity?.toFixed(4) ?? "N/A"}\n`;
    ctx += `- Sensitivity : ${overrideMetrics.sensitivity?.toFixed(4) ?? "N/A"}\n`;
  }

  // ── Live prediction usage stats ───────────────────────────
  if (predictionStats) {
    ctx += "\n### Live Prediction Statistics (from database):\n";
    ctx += `- Total predictions made  : ${predictionStats.totalPredictions}\n`;
    ctx += `- Heart disease detected  : ${predictionStats.positivePredictions} `;
    ctx += `(${predictionStats.positiveRate}% positive rate)\n`;
    ctx += `- No heart disease        : ${predictionStats.negativePredictions}\n`;

    if (Object.keys(predictionStats.perModel).length > 0) {
      ctx += "\nPer-model usage:\n";
      for (const [name, s] of Object.entries(predictionStats.perModel)) {
        ctx += `  • ${name}: ${s.total} predictions, avg confidence ${s.avgProbability}\n`;
      }
    }
  }

  return ctx;
}

/* ============================================================
 * SYSTEM PROMPT
 * ============================================================ */
const SYSTEM_PROMPT = `You are an expert AI assistant with dual expertise in cardiology and machine learning.
You are explaining a heart disease prediction system to medical professionals and researchers.

Guidelines:
- Use accurate medical and statistical terminology while remaining accessible.
- When explaining metrics, always relate them back to real clinical impact.
- For predictions, clearly explain what the confidence score means for that patient.
- Be honest about model limitations and when a doctor's judgment overrides any AI prediction.
- Reference the actual numbers from the provided data – do not invent or round figures.
- If asked about something not covered in the data provided, say so clearly.`;

/* ============================================================
 * POST /api/explainable-ai/ask
 * General Q&A with full DB context injected
 * ============================================================ */
router.post(
  "/ask",
  asyncHandler(async (req, res) => {
    const { question, prediction, probability, modelName } = req.body;

    if (!question?.trim()) {
      return res
        .status(400)
        .json({ success: false, error: "question is required." });
    }

    // Fetch live data from DB
    const [allMetrics, predictionStats] = await Promise.all([
      fetchAllMetricsFromDB(),
      fetchPredictionStats(),
    ]);

    let context = buildMetricsContext(allMetrics, predictionStats);

    // Append the specific prediction that the user is asking about (if any)
    if (prediction !== undefined && probability !== undefined) {
      context += "\n### Prediction Being Discussed:\n";
      context += `- Model        : ${modelName ?? "Not specified"}\n`;
      context += `- Result       : ${prediction === 1 ? "Heart Disease Likely" : "Heart Disease Unlikely"}\n`;
      context += `- Confidence   : ${(probability * 100).toFixed(2)}%\n`;
    }

    try {
      const response = await geminiModel.generateContent([
        { text: `${SYSTEM_PROMPT}\n\n${context}` },
        {
          text: `User question: ${question}\n\nProvide a detailed, accurate response based on the data above.`,
        },
      ]);

      res.json({
        success: true,
        question,
        answer: response.response.text(),
        modelUsed: "gemini-2.5-flash",
        dataSource: {
          modelsInDB: Object.keys(allMetrics).length,
          totalPredictions: predictionStats.totalPredictions,
        },
      });
    } catch (err) {
      console.error("[Gemini] /ask error:", err.message);
      res
        .status(500)
        .json({
          success: false,
          error: "Gemini API call failed.",
          details: err.message,
        });
    }
  }),
);

/* ============================================================
 * POST /api/explainable-ai/explain-metrics
 * Deep explanation of one specific metric with live values
 * ============================================================ */
router.post(
  "/explain-metrics",
  asyncHandler(async (req, res) => {
    const { metricName, modelName } = req.body;

    if (!metricName?.trim()) {
      return res
        .status(400)
        .json({ success: false, error: "metricName is required." });
    }

    // Fetch live data
    const [allMetrics, predictionStats] = await Promise.all([
      fetchAllMetricsFromDB(),
      fetchPredictionStats(),
    ]);

    // If a specific model was requested, surface its metrics prominently
    const targetMetrics = modelName ? (allMetrics[modelName] ?? null) : null;
    const context = buildMetricsContext(
      allMetrics,
      predictionStats,
      targetMetrics,
    );

    try {
      const response = await geminiModel.generateContent([
        { text: `${SYSTEM_PROMPT}\n\n${context}` },
        {
          text: `Explain the "${metricName}" metric${modelName ? ` for the ${modelName} model` : " across all models"}. Cover:
1. What it measures and the mathematical definition
2. Why it matters specifically for heart disease prediction (clinical impact)
3. How to interpret the actual values shown above
4. What an ideal value looks like for this medical use case
5. Any trade-offs with other metrics (e.g. precision vs recall)`,
        },
      ]);

      res.json({
        success: true,
        metric: metricName,
        model: modelName ?? "all",
        explanation: response.response.text(),
        liveMetrics: targetMetrics ?? allMetrics,
      });
    } catch (err) {
      console.error("[Gemini] /explain-metrics error:", err.message);
      res
        .status(500)
        .json({
          success: false,
          error: "Gemini API call failed.",
          details: err.message,
        });
    }
  }),
);

/* ============================================================
 * POST /api/explainable-ai/compare-models
 * AI comparison of all models using live DB metrics
 * ============================================================ */
router.post(
  "/compare-models",
  asyncHandler(async (req, res) => {
    const { modelNames } = req.body; // optional filter

    const [allMetrics, predictionStats] = await Promise.all([
      fetchAllMetricsFromDB(),
      fetchPredictionStats(),
    ]);

    // Filter to requested models if provided
    const metricsToCompare = modelNames?.length
      ? Object.fromEntries(
          Object.entries(allMetrics).filter(([k]) => modelNames.includes(k)),
        )
      : allMetrics;

    if (Object.keys(metricsToCompare).length === 0) {
      return res.status(404).json({
        success: false,
        error: "No metrics found for the requested models. Train models first.",
      });
    }

    const context = buildMetricsContext(metricsToCompare, predictionStats);

    try {
      const response = await geminiModel.generateContent([
        { text: `${SYSTEM_PROMPT}\n\n${context}` },
        {
          text: `Compare ${modelNames?.length ? modelNames.join(", ") : "all available"} models. Include:
1. Which model performs best overall and why (cite the actual metric values)
2. Which model is safest for clinical use (consider recall / false negatives)
3. Trade-offs: speed vs accuracy, interpretability vs performance
4. Specific strengths and weaknesses of each model
5. Your recommendation for deployment in a hospital setting`,
        },
      ]);

      res.json({
        success: true,
        comparison: response.response.text(),
        metrics: metricsToCompare,
        dataSource: {
          modelsCompared: Object.keys(metricsToCompare).length,
          totalPredictions: predictionStats.totalPredictions,
        },
      });
    } catch (err) {
      console.error("[Gemini] /compare-models error:", err.message);
      res
        .status(500)
        .json({
          success: false,
          error: "Gemini API call failed.",
          details: err.message,
        });
    }
  }),
);

/* ============================================================
 * POST /api/explainable-ai/explain-prediction
 * Explain one specific past prediction retrieved from MongoDB
 * ============================================================ */
router.post(
  "/explain-prediction",
  asyncHandler(async (req, res) => {
    const { predictionId, prediction, probability, modelName, features } =
      req.body;

    // Load from DB if an ID is provided, otherwise use request body values
    let predDoc = null;

    if (predictionId) {
      predDoc = await Prediction.findById(predictionId).lean();
      if (!predDoc) {
        return res
          .status(404)
          .json({ success: false, error: "Prediction not found." });
      }
    }

    const usedModel = predDoc?.modelName ?? modelName ?? "unknown";
    const usedPrediction = predDoc?.prediction ?? prediction;
    const usedProbability = predDoc?.probability ?? probability;
    const usedFeatures = predDoc?.inputFeatures ?? features;

    if (usedPrediction === undefined || usedProbability === undefined) {
      return res.status(400).json({
        success: false,
        error:
          "Provide either a predictionId or { prediction, probability, modelName }.",
      });
    }

    // Fetch model metrics from DB
    const [modelMetrics, predictionStats] = await Promise.all([
      fetchModelMetricsFromDB(usedModel),
      fetchPredictionStats(),
    ]);

    const allMetrics = modelMetrics ? { [usedModel]: modelMetrics } : {};
    let context = buildMetricsContext(allMetrics, predictionStats);

    // Add the patient data to the context
    context += `\n### Patient Prediction Details:\n`;
    context += `- Model used    : ${usedModel}\n`;
    context += `- Result        : ${usedPrediction === 1 ? "Heart Disease Likely" : "Heart Disease Unlikely"}\n`;
    context += `- Confidence    : ${(usedProbability * 100).toFixed(2)}%\n`;

    if (usedFeatures) {
      context += `\nPatient input features:\n`;
      const featureMap =
        usedFeatures instanceof Map
          ? Object.fromEntries(usedFeatures)
          : usedFeatures;

      for (const [fname, fval] of Object.entries(featureMap)) {
        context += `  • ${fname}: ${fval}\n`;
      }
    }

    try {
      const response = await geminiModel.generateContent([
        { text: `${SYSTEM_PROMPT}\n\n${context}` },
        {
          text: `Explain this specific patient prediction in plain language. Cover:
1. What the result means for this patient
2. What the ${(usedProbability * 100).toFixed(1)}% confidence score means in practice
3. How the model's reliability (from the metrics above) affects how much to trust this result
4. Which input features are most commonly associated with this outcome
5. What next steps a clinician should consider`,
        },
      ]);

      res.json({
        success: true,
        prediction: usedPrediction,
        probability: usedProbability,
        modelName: usedModel,
        label:
          usedPrediction === 1
            ? "Heart Disease Likely"
            : "Heart Disease Unlikely",
        explanation: response.response.text(),
        metrics: modelMetrics,
      });
    } catch (err) {
      console.error("[Gemini] /explain-prediction error:", err.message);
      res
        .status(500)
        .json({
          success: false,
          error: "Gemini API call failed.",
          details: err.message,
        });
    }
  }),
);

/* ============================================================
 * GET /api/explainable-ai/summary
 * Returns a plain-English AI summary of the whole system's
 * current state (metrics + usage stats) – useful for dashboards
 * ============================================================ */
router.get(
  "/summary",
  asyncHandler(async (_req, res) => {
    const [allMetrics, predictionStats, recentSessions] = await Promise.all([
      fetchAllMetricsFromDB(),
      fetchPredictionStats(),
      TrainingSession.find({ status: "completed" })
        .sort({ endTime: -1 })
        .limit(3)
        .lean(),
    ]);

    const context = buildMetricsContext(allMetrics, predictionStats);

    let trainingCtx = "";
    if (recentSessions.length > 0) {
      trainingCtx += "\n### Recent Retraining Activity:\n";
      for (const s of recentSessions) {
        trainingCtx += `- ${s.modelType} retrained on ${new Date(s.endTime).toLocaleDateString()}`;
        trainingCtx += ` using ${s.originalFilename ?? "unknown file"}`;
        trainingCtx += ` (${s.numSamples ?? "?"} samples, ${s.numFeatures ?? "?"} features)\n`;
      }
    }

    try {
      const response = await geminiModel.generateContent([
        { text: `${SYSTEM_PROMPT}\n\n${context}${trainingCtx}` },
        {
          text: `Write a concise executive summary (3-5 paragraphs) of this heart disease prediction system's current performance and status. 
Include: best-performing model, overall system reliability, prediction volume, and any notable observations.
Write it as if presenting to a hospital board — clear, factual, and professional.`,
        },
      ]);

      res.json({
        success: true,
        summary: response.response.text(),
        snapshot: {
          models: Object.keys(allMetrics),
          totalPredictions: predictionStats.totalPredictions,
          positiveRate: predictionStats.positiveRate + "%",
          lastRetrained: recentSessions[0]?.endTime ?? null,
        },
      });
    } catch (err) {
      console.error("[Gemini] /summary error:", err.message);
      res
        .status(500)
        .json({
          success: false,
          error: "Gemini API call failed.",
          details: err.message,
        });
    }
  }),
);

export default router;
