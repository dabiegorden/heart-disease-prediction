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

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 5000;

// ---------------------------------------------------------
// Middleware
// ---------------------------------------------------------
app.use(morgan("dev"));
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:3000",
    credentials: true,
  })
);

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// ---------------------------------------------------------
// Initialize ONNX Models
// ---------------------------------------------------------
const modelsPath =
  process.env.MODELS_PATH || path.join(__dirname, "../src/models");

const modelLoader = new ModelLoader(modelsPath);
let modelsReady = false;

async function initializeModels() {
  console.log("🚀 Initializing ONNX model loader...");

  await modelLoader.loadAllModels();
  modelLoader.loadMetrics();
  modelLoader.loadScaler();

  const loadedModels = modelLoader.getAvailableModels();

  if (loadedModels.length === 0) {
    console.error("❌ No ONNX models found — prediction endpoints will fail.");
  } else {
    console.log(`✓ Loaded ${loadedModels.length} ONNX models:`, loadedModels);
  }

  modelsReady = true;

  console.log("✓ All models & scaler initialized.\n");
}

await initializeModels();

// ---------------------------------------------------------
// Health Check Endpoint
// ---------------------------------------------------------
app.get("/health", (req, res) => {
  res.json({
    status: modelsReady ? "ok" : "initializing",
    modelsReady,
    modelsLoaded: modelLoader.getAvailableModels(),
    scalerLoaded: !!modelLoader.scalerStats,
    timestamp: new Date().toISOString(),
  });
});

// ---------------------------------------------------------
// Prediction API Routes
// ---------------------------------------------------------
app.use("/api/predict", createPredictRouter(modelLoader));

// ---------------------------------------------------------
// API Information Endpoint
// ---------------------------------------------------------
app.get("/api/info", (req, res) => {
  res.json({
    name: "Heart Disease Prediction API",
    version: "1.0.0",
    models: modelLoader.getAvailableModels(),
    features: 12,
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

// ---------------------------------------------------------
// 404 Handler
// ---------------------------------------------------------
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: "Endpoint Not Found",
    path: req.path,
  });
});

// ---------------------------------------------------------
// Global Error Handler
// ---------------------------------------------------------
app.use(errorHandler);

// ---------------------------------------------------------
// Start Server
// ---------------------------------------------------------
app.listen(PORT, () => {
  const loadedCount = modelLoader.getAvailableModels().length;
  const scalerStatus = modelLoader.scalerStats ? "Yes" : "No";

  console.log(`
╔══════════════════════════════════════════════════╗
║   🏥 HEART DISEASE PREDICTION API
║
║   ➤ Server Running: http://localhost:${PORT}
║   ➤ Models Loaded: ${loadedCount}
║   ➤ Scaler Loaded: ${scalerStatus}
║
╚══════════════════════════════════════════════════╝
`);
});
