import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Models that require 3D input: [1, 1, features]
 */
const DEEP_MODELS = new Set(["cnn", "cnn_lstm"]);

/* ============================================================
 * SAFE SERIALIZER (BigInt + TypedArray → JSON-safe)
 * ============================================================ */
function serialize(value) {
  if (typeof value === "bigint") {
    return Number(value);
  }

  if (Array.isArray(value)) {
    return value.map(serialize);
  }

  if (ArrayBuffer.isView(value)) {
    return Array.from(value, serialize);
  }

  if (value && typeof value === "object") {
    const obj = {};
    for (const key in value) {
      obj[key] = serialize(value[key]);
    }
    return obj;
  }

  return value;
}

class ModelLoader {
  constructor(modelsPath) {
    this.modelsPath = modelsPath;
    this.models = new Map();
    this.metrics = null;
    this.scalerStats = null;
  }

  /* ============================================================
   * LOAD ALL ONNX MODELS
   * ============================================================ */
  async loadAllModels() {
    try {
      const modelFiles = fs
        .readdirSync(this.modelsPath)
        .filter((file) => file.endsWith(".onnx"));

      console.log(`📦 Loading ${modelFiles.length} ONNX models...\n`);

      for (const file of modelFiles) {
        const modelName = file.replace(".onnx", "");
        const modelPath = path.join(this.modelsPath, file);

        try {
          const session = await ort.InferenceSession.create(modelPath);
          this.models.set(modelName, session);
          console.log(`  ✓ Loaded: ${modelName}`);
        } catch (err) {
          console.error(`  ✗ Failed to load ${modelName}: ${err.message}`);
        }
      }

      console.log(`\n✓ Loaded ${this.models.size} models`);
      return true;
    } catch (err) {
      console.error("❌ Model loading failed:", err);
      return false;
    }
  }

  /* ============================================================
   * LOAD METRICS
   * ============================================================ */
  loadMetrics() {
    const file = path.join(this.modelsPath, "model_metrics.json");

    if (!fs.existsSync(file)) {
      console.warn("⚠ model_metrics.json not found");
      return null;
    }

    this.metrics = JSON.parse(fs.readFileSync(file, "utf-8"));
    console.log("✓ Model metrics loaded");
    return this.metrics;
  }

  /* ============================================================
   * LOAD SCALER
   * ============================================================ */
  loadScaler() {
    const file = path.join(this.modelsPath, "scaler.json");

    if (!fs.existsSync(file)) {
      console.warn("⚠ scaler.json not found");
      return null;
    }

    this.scalerStats = JSON.parse(fs.readFileSync(file, "utf-8"));
    console.log("✓ Scaler loaded");
    return this.scalerStats;
  }

  /* ============================================================
   * APPLY STANDARD SCALER
   * ============================================================ */
  applyScaling(features) {
    if (!this.scalerStats) return features;

    const { mean, scale } = this.scalerStats;
    return features.map((v, i) => (v - mean[i]) / (scale[i] || 1));
  }

  /* ============================================================
   * METADATA
   * ============================================================ */
  getAvailableModels() {
    return Array.from(this.models.keys());
  }

  getMetrics(modelName) {
    return this.metrics?.[modelName] || null;
  }

  getAllMetrics() {
    return this.metrics || {};
  }

  /* ============================================================
   * RUN PREDICTION (ML + DL)
   * ============================================================ */
  async predict(modelName, features) {
    const session = this.models.get(modelName);
    if (!session) {
      throw new Error(`Model not found: ${modelName}`);
    }

    try {
      const numeric = features.map(Number);
      const scaled = this.applyScaling(numeric);

      const inputName = session.inputNames[0];

      const inputShape = DEEP_MODELS.has(modelName)
        ? [1, 1, scaled.length] // CNN / CNN-LSTM
        : [1, scaled.length]; // Traditional ML

      const tensor = new ort.Tensor(
        "float32",
        Float32Array.from(scaled),
        inputShape
      );

      const outputs = await session.run({ [inputName]: tensor });

      const outputKey = Object.keys(outputs)[0];
      const rawData = outputs[outputKey].data;

      let prediction = null;
      let probability = null;

      if (rawData.length === 1) {
        probability = Number(rawData[0]);
        prediction = probability >= 0.5 ? 1 : 0;
      } else if (rawData.length >= 2) {
        probability = Number(rawData[1]);
        prediction = probability >= 0.5 ? 1 : 0;
      }

      return serialize({
        model: modelName,
        prediction,
        probability,
        rawOutput: rawData,
        inputShape,
      });
    } catch (err) {
      throw new Error(`Inference failed (${modelName}): ${err.message}`);
    }
  }
}

export default ModelLoader;
