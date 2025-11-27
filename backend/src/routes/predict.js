/**
 * Prediction Routes for Heart Disease ML API
 */

import express from "express";
import { asyncHandler } from "../middleware/errorHandler.js";

const EXPECTED_FEATURE_COUNT = 12;

export function createPredictRouter(modelLoader) {
  const router = express.Router();

  /**
   * Validate raw feature input
   */
  function validateFeatures(features) {
    if (!Array.isArray(features)) {
      return "Features must be an array.";
    }

    if (features.length !== EXPECTED_FEATURE_COUNT) {
      return `Expected ${EXPECTED_FEATURE_COUNT} features, received ${features.length}.`;
    }

    const invalid = features.some(
      (v) => v === null || v === undefined || isNaN(Number(v))
    );

    if (invalid) {
      return "All features must be valid numeric values.";
    }

    return null;
  }

  /**
   * GET /api/models
   * Return list of available ONNX models + metrics
   */
  router.get(
    "/models",
    asyncHandler(async (req, res) => {
      const models = modelLoader.getAvailableModels();
      const metrics = modelLoader.getAllMetrics();

      res.json({
        success: true,
        count: models.length,
        models,
        metrics,
      });
    })
  );

  /**
   * POST /api/predict/compare
   * Run all ONNX models and return combined predictions
   */
  router.post(
    "/compare",
    asyncHandler(async (req, res) => {
      const { features } = req.body;

      console.log("[API] /compare → received features:", features);

      // Validate features
      const validationError = validateFeatures(features);
      if (validationError) {
        console.log("[API] Validation failed:", validationError);
        return res.status(400).json({ success: false, error: validationError });
      }

      const availableModels = modelLoader.getAvailableModels();
      const predictions = {};

      console.log(
        `[API] Running inference on ${availableModels.length} ONNX models...\n`
      );

      for (const modelName of availableModels) {
        try {
          const result = await modelLoader.predict(modelName, features);

          predictions[modelName] = {
            prediction: result.prediction,
            probability: result.probability ?? null,
            metrics: modelLoader.getMetrics(modelName),
          };

          console.log(`[API] ✔ ${modelName} OK`, result);
        } catch (err) {
          console.error(`[API] ✗ ${modelName} ERROR →`, err.message);

          predictions[modelName] = {
            error: `Prediction failed: ${err.message}`,
          };
        }
      }

      res.json({
        success: true,
        modelCount: availableModels.length,
        predictions,
      });
    })
  );

  /**
   * POST /api/predict/:modelName
   * Run a specific ONNX model
   */
  router.post(
    "/:modelName",
    asyncHandler(async (req, res) => {
      const { modelName } = req.params;
      const { features } = req.body;

      const availableModels = modelLoader.getAvailableModels();

      // Validate model
      if (!availableModels.includes(modelName)) {
        return res.status(404).json({
          success: false,
          error: `Model '${modelName}' not found. Available models: ${availableModels.join(
            ", "
          )}`,
        });
      }

      // Validate features
      const validationError = validateFeatures(features);
      if (validationError) {
        return res.status(400).json({ success: false, error: validationError });
      }

      // Run prediction
      try {
        const result = await modelLoader.predict(modelName, features);

        res.json({
          success: true,
          model: modelName,
          prediction: result.prediction,
          probability: result.probability ?? null,
          metrics: modelLoader.getMetrics(modelName),
        });
      } catch (err) {
        console.error(`[API] Error in /predict/${modelName}:`, err.message);

        res.status(500).json({
          success: false,
          error: err.message,
        });
      }
    })
  );

  return router;
}
