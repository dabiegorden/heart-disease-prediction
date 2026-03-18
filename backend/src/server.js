/**
 * server.js  –  Heart Disease Prediction API
 * ===========================================
 * Changes from original
 * ---------------------
 * 1. /api/info reads featureNames + featureCount dynamically from the loaded
 *    scaler so it stays correct after any retraining run.
 * 2. connectDB is called before app.listen so DB errors surface early.
 * 3. ModelMetrics collection is seeded with the initial model_metrics.json
 *    on first boot (if no records exist yet).
 */

import express from "express";
import cors from "cors";
import morgan from "morgan";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import fs from "fs";

import ModelLoader from "./utils/modelLoader.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { createPredictRouter } from "./routes/predict.js";
import createFederatedRouter from "./routes/federated.js";
import createRetrainRouter from "./routes/retrain.js";
import explainableAIRouter from "./routes/explainable-ai.js";
import { connectDB } from "./config/mongodb.js";
import { ModelMetrics } from "./db/schema.js";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 5000;

/* ============================================================
 * MIDDLEWARE
 * ============================================================ */
app.use(morgan("dev"));
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:3000",
    credentials: true,
  }),
);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

/* ============================================================
 * MODEL INITIALIZATION
 * ============================================================ */
const modelsPath =
  process.env.MODELS_PATH || path.join(__dirname, "../src/models");
const modelLoader = new ModelLoader(modelsPath);
let modelsReady = false;

async function initializeModels() {
  console.log("🚀 Initializing ONNX model loader…");

  await modelLoader.loadAllModels();
  modelLoader.loadMetrics();
  modelLoader.loadScaler();

  const available = modelLoader.getAvailableModels();

  if (available.length === 0) {
    console.error("❌ No ONNX models loaded — predictions will fail.");
  } else {
    console.log(
      `✓ Loaded ${available.length} model(s): ${available.join(", ")}`,
    );
  }

  modelsReady = true;
  console.log("✓ Model initialization complete.\n");
}

/**
 * On first boot, seed the ModelMetrics collection from model_metrics.json
 * so the /comparison endpoint has data without requiring a retrain.
 */
async function seedInitialMetrics() {
  try {
    const count = await ModelMetrics.countDocuments({});
    if (count > 0) return; // already seeded

    const metricsFile = path.join(modelsPath, "model_metrics.json");
    if (!fs.existsSync(metricsFile)) return;

    const raw = JSON.parse(fs.readFileSync(metricsFile, "utf-8"));
    const records = Object.entries(raw).map(([modelName, m]) => ({
      modelName,
      source: "initial",
      accuracy: m.accuracy,
      precision: m.precision,
      recall: m.recall,
      f1: m.f1,
      f1_score: m.f1_score,
      auc: m.auc,
      auc_roc: m.auc_roc,
      cv_mean: m.cv_mean,
      cv_std: m.cv_std,
      confusion_matrix: m.confusion_matrix,
      trainedAt: new Date(),
    }));

    await ModelMetrics.insertMany(records, { ordered: false });
    console.log(
      `✓ Seeded ${records.length} initial model metrics into MongoDB`,
    );
  } catch (err) {
    console.warn("⚠  Could not seed initial metrics:", err.message);
  }
}

/* ============================================================
 * HEALTH CHECK
 * ============================================================ */
app.get("/health", (_req, res) => {
  res.json({
    status: modelsReady ? "ok" : "initializing",
    modelsReady,
    modelsLoaded: modelLoader.getAvailableModels(),
    scalerLoaded: !!modelLoader.scalerStats,
    timestamp: new Date().toISOString(),
  });
});

/* ============================================================
 * ROUTES
 * ============================================================ */
app.use("/api/predict", createPredictRouter(modelLoader));
app.use("/api/federated", createFederatedRouter());
app.use("/api/retrain", createRetrainRouter());
app.use("/api/explainable-ai", explainableAIRouter);

/* ============================================================
 * API INFO  –  feature list is read from the loaded scaler
 * ============================================================ */
app.get("/api/info", (_req, res) => {
  const models = modelLoader.getAvailableModels();

  res.json({
    name: "Heart Disease Prediction API",
    version: "1.2.0",
    models: models.map((name) => ({
      name,
      type: name.includes("cnn") ? "deep-learning" : "machine-learning",
    })),
    featureCount: modelLoader.getFeatureCount(),
    featureNames: modelLoader.getFeatureNames(),
  });
});

/* ============================================================
 * 404 / ERROR HANDLERS
 * ============================================================ */
app.use((req, res) => {
  res
    .status(404)
    .json({ success: false, error: "Endpoint not found", path: req.path });
});

app.use(errorHandler);

/* ============================================================
 * START
 * ============================================================ */
await connectDB();
await initializeModels();
await seedInitialMetrics();

app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════╗
║   🏥 HEART DISEASE PREDICTION API
║
║   ➤ Server  : http://localhost:${PORT}
║   ➤ Models  : ${modelLoader.getAvailableModels().length} loaded
║   ➤ Features: ${modelLoader.getFeatureCount()} (${modelLoader.getFeatureNames().join(", ")})
║   ➤ Scaler  : ${modelLoader.scalerStats ? "✓" : "✗ not loaded"}
║
╚══════════════════════════════════════════════════╝
`);
});
