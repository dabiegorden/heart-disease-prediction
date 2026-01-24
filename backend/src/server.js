/**
 * Express Server – Heart Disease Prediction API
 */

import express from "express";
import cors from "cors";
import morgan from "morgan";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

import ModelLoader from "./utils/modelLoader.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { createPredictRouter } from "./routes/predict.js";
import federatedRouter from "./routes/federated.js";
import createRetrainRouter from "./routes/retrain.js";
import explainableAIRouter from "./routes/explainable-ai.js";
import { connectDB } from "./config/mongodb.js";

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
  console.log("🚀 Initializing ONNX model loader...");

  const loaded = await modelLoader.loadAllModels();
  modelLoader.loadMetrics();
  modelLoader.loadScaler();

  const models = modelLoader.getAvailableModels();

  if (!loaded || models.length === 0) {
    console.error("❌ No ONNX models loaded — predictions will fail.");
  } else {
    console.log(`✓ Loaded ${models.length} models:`, models);
  }

  modelsReady = true;
  console.log("✓ Model initialization complete.\n");
}

await initializeModels();

/* ============================================================
 * HEALTH CHECK
 * ============================================================ */
app.get("/health", (req, res) => {
  res.json({
    status: modelsReady ? "ok" : "initializing",
    modelsReady,
    modelsLoaded: modelLoader.getAvailableModels(),
    scalerLoaded: !!modelLoader.scalerStats,
    timestamp: new Date().toISOString(),
  });
});

/* ============================================================
 * PREDICTION ROUTES
 * ============================================================ */
app.use("/api/predict", createPredictRouter(modelLoader));
app.use("/api/federated", federatedRouter);
app.use("/api/retrain", createRetrainRouter());
app.use("/api/explainable-ai", explainableAIRouter);

/* ============================================================
 * API INFO
 * ============================================================ */
app.get("/api/info", (req, res) => {
  const models = modelLoader.getAvailableModels();

  const modelDetails = models.map((name) => ({
    name,
    type: name.includes("cnn") ? "deep-learning" : "machine-learning",
  }));

  res.json({
    name: "Heart Disease Prediction API",
    version: "1.1.0",
    models: modelDetails,
    featureCount: 12,
    featureNames: [
      "age",
      "sex",
      "chestpaintype",
      "restingbps",
      "cholesterol",
      "fastingbloodsugar",
      "restingecg",
      "maxheartrate",
      "exerciseangina",
      "oldpeak",
      "slope",
      "noofmajorvessels",
    ],
  });
});

/* ============================================================
 * 404 HANDLER
 * ============================================================ */
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: "Endpoint Not Found",
    path: req.path,
  });
});

/* ============================================================
 * GLOBAL ERROR HANDLER
 * ============================================================ */
app.use(errorHandler);

/* ============================================================
 * START SERVER
 * ============================================================ */
app.listen(PORT, async () => {
  await connectDB();
  console.log(`
╔══════════════════════════════════════════════════╗
║   🏥 HEART DISEASE PREDICTION API
║
║   ➤ Server: http://localhost:${PORT}
║   ➤ Models Loaded: ${modelLoader.getAvailableModels().length}
║   ➤ Scaler Loaded: ${modelLoader.scalerStats ? "Yes" : "No"}
║
╚══════════════════════════════════════════════════╝
`);
});
