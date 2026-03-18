/**
 * routes/predict.js
 * =================
 * Prediction endpoints.
 *
 * Changes from original
 * ---------------------
 * 1. Feature count + names are read from the loaded scaler (dynamic) so
 *    the route works after retraining on any dataset column set.
 * 2. Every prediction (single-model and compare) is persisted to MongoDB.
 * 3. Features can be submitted as an ordered array OR as a name→value object.
 */

import express from "express";
import { asyncHandler } from "../middleware/errorHandler.js";
import { Prediction } from "../db/schema.js";

export function createPredictRouter(modelLoader) {
  const router = express.Router();

  /* ----------------------------------------------------------
   * Validate incoming features against the scaler's feature list.
   * Accepts:
   *   - Array  → must have exactly featureCount elements
   *   - Object → keys are mapped to canonical feature names
   * ---------------------------------------------------------- */
  function validateFeatures(features) {
    if (features === null || features === undefined) {
      return "Features are required.";
    }

    const featureCount = modelLoader.getFeatureCount();
    const featureNames = modelLoader.getFeatureNames();

    if (Array.isArray(features)) {
      if (features.length !== featureCount) {
        return (
          `Expected ${featureCount} features (${featureNames.join(", ")}), ` +
          `but received ${features.length}.`
        );
      }
      const invalid = features.some(
        (v) => v === null || v === undefined || isNaN(Number(v)),
      );
      if (invalid) return "All feature values must be valid numbers.";
    } else if (typeof features === "object") {
      const missing = featureNames.filter(
        (name) => features[name] === undefined || features[name] === null,
      );
      if (missing.length > 0) {
        return `Missing feature(s): ${missing.join(", ")}`;
      }
      const invalid = featureNames.some((name) =>
        isNaN(Number(features[name])),
      );
      if (invalid) return "All feature values must be valid numbers.";
    } else {
      return "Features must be an array or an object.";
    }

    return null; // valid
  }

  /* ----------------------------------------------------------
   * Helper: persist a prediction to MongoDB (non-blocking)
   * ---------------------------------------------------------- */
  async function savePrediction(data) {
    try {
      await Prediction.create(data);
    } catch (err) {
      // Never let a DB error break the API response
      console.error("[MongoDB] Failed to save prediction:", err.message);
    }
  }

  /* ============================================================
   * GET /api/predict/models
   * List available models + their metrics
   * ============================================================ */
  router.get(
    "/models",
    asyncHandler(async (req, res) => {
      res.json({
        success: true,
        count: modelLoader.getAvailableModels().length,
        models: modelLoader.getAvailableModels(),
        featureCount: modelLoader.getFeatureCount(),
        featureNames: modelLoader.getFeatureNames(),
        metrics: modelLoader.getAllMetrics(),
      });
    }),
  );

  /* ============================================================
   * POST /api/predict/compare
   * Run all models and return side-by-side results
   * ============================================================ */
  router.post(
    "/compare",
    asyncHandler(async (req, res) => {
      const { features } = req.body;

      const validationError = validateFeatures(features);
      if (validationError) {
        return res.status(400).json({ success: false, error: validationError });
      }

      const models = modelLoader.getAvailableModels();
      const predictions = {};

      for (const modelName of models) {
        try {
          const result = await modelLoader.predict(modelName, features);
          predictions[modelName] = {
            prediction: result.prediction,
            probability: result.probability,
            label:
              result.prediction === 1
                ? "Heart Disease Likely"
                : "Heart Disease Unlikely",
            inputShape: result.inputShape,
            metrics: modelLoader.getMetrics(modelName),
          };
        } catch (err) {
          predictions[modelName] = { error: err.message };
        }
      }

      // Persist the compare request (store all results in allModelResults)
      savePrediction({
        modelName: "compare",
        inputFeatures: Array.isArray(features)
          ? Object.fromEntries(
              modelLoader.getFeatureNames().map((n, i) => [n, features[i]]),
            )
          : features,
        scaledFeatures: modelLoader.applyScaling(features),
        prediction: 0, // placeholder – no single prediction for compare
        probability: 0,
        label: "Heart Disease Unlikely",
        allModelResults: predictions,
        requestIp: req.ip,
        userAgent: req.headers["user-agent"],
      });

      res.json({
        success: true,
        modelCount: models.length,
        predictions,
      });
    }),
  );

  /* ============================================================
   * POST /api/predict/:modelName
   * Single-model prediction
   * ============================================================ */
  router.post(
    "/:modelName",
    asyncHandler(async (req, res) => {
      const { modelName } = req.params;
      const { features } = req.body;
      const models = modelLoader.getAvailableModels();

      if (!models.includes(modelName)) {
        return res.status(404).json({
          success: false,
          error: `Model '${modelName}' not found.`,
          availableModels: models,
        });
      }

      const validationError = validateFeatures(features);
      if (validationError) {
        return res.status(400).json({ success: false, error: validationError });
      }

      try {
        const result = await modelLoader.predict(modelName, features);
        const label =
          result.prediction === 1
            ? "Heart Disease Likely"
            : "Heart Disease Unlikely";

        // Persist to MongoDB
        savePrediction({
          modelName,
          inputFeatures: Array.isArray(features)
            ? Object.fromEntries(
                modelLoader.getFeatureNames().map((n, i) => [n, features[i]]),
              )
            : features,
          scaledFeatures: result.scaledInput,
          prediction: result.prediction,
          probability: result.probability,
          label,
          requestIp: req.ip,
          userAgent: req.headers["user-agent"],
        });

        res.json({
          success: true,
          model: modelName,
          prediction: result.prediction,
          probability: result.probability,
          label,
          inputShape: result.inputShape,
          metrics: modelLoader.getMetrics(modelName),
        });
      } catch (err) {
        res.status(500).json({ success: false, error: err.message });
      }
    }),
  );

  /* ============================================================
   * GET /api/predict/history
   * Retrieve past predictions from MongoDB
   * ============================================================ */
  router.get(
    "/history",
    asyncHandler(async (req, res) => {
      const limit = Math.min(parseInt(req.query.limit ?? "50", 10), 200);
      const skip = Math.max(parseInt(req.query.skip ?? "0", 10), 0);
      const model = req.query.model;

      const filter = model ? { modelName: model } : {};

      const [records, total] = await Promise.all([
        Prediction.find(filter)
          .sort({ createdAt: -1 })
          .skip(skip)
          .limit(limit)
          .lean(),
        Prediction.countDocuments(filter),
      ]);

      res.json({
        success: true,
        total,
        returned: records.length,
        skip,
        limit,
        predictions: records,
      });
    }),
  );

  return router;
}
