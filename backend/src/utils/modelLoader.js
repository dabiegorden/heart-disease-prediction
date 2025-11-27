import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

class ModelLoader {
  constructor(modelsPath) {
    this.modelsPath = modelsPath;
    this.models = new Map();
    this.metrics = null;
    this.scalerStats = null;
  }

  /**
   * Load ONNX models
   */
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

      console.log(`\n✓ Successfully loaded ${this.models.size} models`);
      return true;
    } catch (err) {
      console.error("❌ Failed to load models:", err);
      return false;
    }
  }

  /**
   * Load model metrics
   */
  loadMetrics() {
    const metricsFile = path.join(this.modelsPath, "model_metrics.json");

    if (!fs.existsSync(metricsFile)) {
      console.warn("⚠ No model_metrics.json found");
      return null;
    }

    try {
      this.metrics = JSON.parse(fs.readFileSync(metricsFile, "utf-8"));
      console.log("✓ Model metrics loaded");
      return this.metrics;
    } catch (err) {
      console.warn("⚠ Error loading metrics:", err.message);
      return null;
    }
  }

  /**
   * Load StandardScaler stats
   */
  loadScaler() {
    const scalerFile = path.join(this.modelsPath, "scaler.json");

    if (!fs.existsSync(scalerFile)) {
      console.warn("⚠ No scaler.json found — using raw input");
      return null;
    }

    try {
      this.scalerStats = JSON.parse(fs.readFileSync(scalerFile, "utf-8"));
      console.log("✓ Scaler stats loaded");
      return this.scalerStats;
    } catch (err) {
      console.error("❌ Failed to load scaler:", err.message);
      return null;
    }
  }

  /**
   * Apply StandardScaler transform
   */
  applyScaling(features) {
    if (!this.scalerStats) return features;

    const { mean, scale } = this.scalerStats;

    return features.map((value, index) => {
      const m = mean[index];
      const s = scale[index] || 1;
      return (value - m) / s;
    });
  }

  /**
   * Return available ONNX model names
   */
  getAvailableModels() {
    return Array.from(this.models.keys());
  }

  /**
   * Return metrics for one model
   */
  getMetrics(modelName) {
    return this.metrics?.[modelName] || null;
  }

  getAllMetrics() {
    return this.metrics || {};
  }

  /**
   * Predict using a loaded ONNX model
   */
  async predict(modelName, features) {
    const session = this.models.get(modelName);

    if (!session) {
      throw new Error(`Model not found: ${modelName}`);
    }

    try {
      // Convert inputs
      const numeric = features.map((v) => Number(v));
      const scaled = this.applyScaling(numeric);

      // Create input tensor
      const inputName = session.inputNames[0];
      const tensor = new ort.Tensor("float32", Float32Array.from(scaled), [
        1,
        scaled.length,
      ]);

      // Run inference
      const outputs = await session.run({ [inputName]: tensor });

      // Extract prediction
      if (!outputs["label"]) {
        throw new Error(`ONNX output missing 'label' for model ${modelName}`);
      }

      const prediction = Number(outputs["label"].data[0]);

      // Extract probability tensor if available
      const probability = outputs["probabilities"]
        ? Array.from(outputs["probabilities"].data)
        : null;

      return {
        model: modelName,
        prediction,
        probability,
        rawOutputs: Object.keys(outputs),
      };
    } catch (err) {
      throw new Error(`Inference failed for ${modelName}: ${err.message}`);
    }
  }
}

export default ModelLoader;
