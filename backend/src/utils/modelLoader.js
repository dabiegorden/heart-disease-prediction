/**
 * utils/modelLoader.js
 * ====================
 * Loads all ONNX models and applies the scaler produced by the Python
 * training pipeline.
 *
 * Key change from the original
 * ----------------------------
 * The Python preprocessor now saves `feature_names` inside scaler.json.
 * The loader reads that list so it always knows the correct feature order
 * and count – even after retraining on a dataset with different columns.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Models that need a 3-D input tensor  [batch, 1, features] */
const DEEP_MODELS = new Set(["cnn", "cnn_lstm"]);

/* ============================================================
 * SAFE SERIALIZER  (BigInt + TypedArray → plain JSON values)
 * ============================================================ */
function serialize(value) {
  if (typeof value === "bigint") return Number(value);
  if (Array.isArray(value)) return value.map(serialize);
  if (ArrayBuffer.isView(value)) return Array.from(value, serialize);
  if (value && typeof value === "object") {
    const out = {};
    for (const k in value) out[k] = serialize(value[k]);
    return out;
  }
  return value;
}

/* ============================================================
 * MODEL LOADER
 * ============================================================ */
class ModelLoader {
  constructor(modelsPath) {
    this.modelsPath = modelsPath;
    this.models = new Map(); // modelName → InferenceSession
    this.metrics = null; // model_metrics.json contents
    this.scalerStats = null; // scaler.json contents
  }

  /* ----------------------------------------------------------
   * Load all .onnx files in modelsPath
   * ---------------------------------------------------------- */
  async loadAllModels() {
    try {
      const onnxFiles = fs
        .readdirSync(this.modelsPath)
        .filter((f) => f.endsWith(".onnx"));

      console.log(`📦 Loading ${onnxFiles.length} ONNX model(s)…\n`);

      for (const file of onnxFiles) {
        const name = file.replace(".onnx", "");
        const fullPath = path.join(this.modelsPath, file);
        try {
          const session = await ort.InferenceSession.create(fullPath);
          this.models.set(name, session);
          console.log(`  ✓ ${name}`);
        } catch (err) {
          console.error(`  ✗ ${name}: ${err.message}`);
        }
      }

      console.log(`\n✓ ${this.models.size} model(s) ready`);
      return true;
    } catch (err) {
      console.error("❌ loadAllModels failed:", err.message);
      return false;
    }
  }

  /* ----------------------------------------------------------
   * Load model_metrics.json
   * ---------------------------------------------------------- */
  loadMetrics() {
    const filePath = path.join(this.modelsPath, "model_metrics.json");
    if (!fs.existsSync(filePath)) {
      console.warn("⚠  model_metrics.json not found");
      return null;
    }
    this.metrics = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    console.log("✓ Model metrics loaded");
    return this.metrics;
  }

  /* ----------------------------------------------------------
   * Load scaler.json  (mean, scale, feature_names)
   * ---------------------------------------------------------- */
  loadScaler() {
    const filePath = path.join(this.modelsPath, "scaler.json");
    if (!fs.existsSync(filePath)) {
      console.warn("⚠  scaler.json not found – inputs will NOT be scaled");
      return null;
    }
    this.scalerStats = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const n =
      this.scalerStats.feature_names?.length ?? this.scalerStats.mean?.length;
    console.log(
      `✓ Scaler loaded  (${n} features: ${this.scalerStats.feature_names?.join(", ")})`,
    );
    return this.scalerStats;
  }

  /* ----------------------------------------------------------
   * Metadata helpers
   * ---------------------------------------------------------- */
  getAvailableModels() {
    return Array.from(this.models.keys());
  }
  getMetrics(name) {
    return this.metrics?.[name] ?? null;
  }
  getAllMetrics() {
    return this.metrics ?? {};
  }

  /**
   * The canonical feature list as saved by the Python preprocessor.
   * Falls back to a generic numbered list if scaler.json has no names.
   */
  getFeatureNames() {
    return (
      this.scalerStats?.feature_names ??
      Array.from(
        { length: this.scalerStats?.mean?.length ?? 12 },
        (_, i) => `feature_${i}`,
      )
    );
  }

  getFeatureCount() {
    return this.scalerStats?.mean?.length ?? 12;
  }

  /* ----------------------------------------------------------
   * Standard-scale a raw feature vector.
   * Accepts either an ordered array or a key→value object.
   * ---------------------------------------------------------- */
  applyScaling(input) {
    if (!this.scalerStats)
      return Array.isArray(input) ? input : Object.values(input);

    const { mean, scale, feature_names } = this.scalerStats;

    let ordered;

    if (Array.isArray(input)) {
      // Assume caller already ordered the values correctly
      ordered = input.map(Number);
    } else {
      // Object: reorder by feature_names, fill missing with column mean
      ordered = feature_names.map((name, i) => {
        const val = input[name];
        return val !== undefined && val !== null ? Number(val) : mean[i];
      });
    }

    return ordered.map((v, i) => (v - mean[i]) / (scale[i] || 1));
  }

  /* ----------------------------------------------------------
   * Run inference
   * Accepts features as an ordered Array or as a name→value Object
   * ---------------------------------------------------------- */
  async predict(modelName, features) {
    const session = this.models.get(modelName);
    if (!session) throw new Error(`Model not found: ${modelName}`);

    try {
      const scaled = this.applyScaling(features);
      const inputName = session.inputNames[0];
      const inputShape = DEEP_MODELS.has(modelName)
        ? [1, 1, scaled.length]
        : [1, scaled.length];

      const tensor = new ort.Tensor(
        "float32",
        Float32Array.from(scaled),
        inputShape,
      );

      const outputs = await session.run({ [inputName]: tensor });
      const outputKey = Object.keys(outputs)[0];
      const rawData = outputs[outputKey].data;

      let prediction, probability;

      if (rawData.length === 1) {
        probability = Number(rawData[0]);
        prediction = probability >= 0.5 ? 1 : 0;
      } else {
        probability = Number(rawData[1]);
        prediction = probability >= 0.5 ? 1 : 0;
      }

      return serialize({
        model: modelName,
        prediction,
        probability,
        rawOutput: rawData,
        inputShape,
        scaledInput: scaled,
      });
    } catch (err) {
      throw new Error(`Inference failed (${modelName}): ${err.message}`);
    }
  }
}

export default ModelLoader;
