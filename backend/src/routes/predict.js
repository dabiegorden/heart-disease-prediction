/**
 * Prediction Routes for Heart Disease ML API
 */

import express from "express";
import { asyncHandler } from "../middleware/errorHandler.js";

const EXPECTED_FEATURE_COUNT = 12;

export function createPredictRouter(modelLoader) {
  const router = express.Router();

  /* ============================================================
   * VALIDATION
   * ============================================================ */
  function validateFeatures(features) {
    if (!Array.isArray(features)) {
      return "Features must be an array.";
    }

    if (features.length !== EXPECTED_FEATURE_COUNT) {
      return `Expected ${EXPECTED_FEATURE_COUNT} features, received ${features.length}.`;
    }

    const invalid = features.some(
      (v) => v === null || v === undefined || isNaN(Number(v)),
    );

    if (invalid) {
      return "All features must be valid numeric values.";
    }

    return null;
  }

  /* ============================================================
   * GET /api/predict/models
   * ============================================================ */
  router.get(
    "/models",
    asyncHandler(async (req, res) => {
      res.json({
        success: true,
        count: modelLoader.getAvailableModels().length,
        models: modelLoader.getAvailableModels(),
        metrics: modelLoader.getAllMetrics(),
      });
    }),
  );

  /* ============================================================
   * POST /api/predict/compare
   * ============================================================ */
  router.post(
    "/compare",
    asyncHandler(async (req, res) => {
      const { features } = req.body;

      const validationError = validateFeatures(features);
      if (validationError) {
        return res.status(400).json({
          success: false,
          error: validationError,
        });
      }

      const models = modelLoader.getAvailableModels();
      const predictions = {};

      for (const modelName of models) {
        try {
          const result = await modelLoader.predict(modelName, features);

          predictions[modelName] = {
            prediction: result.prediction,
            probability: result.probability,
            inputShape: result.inputShape,
            metrics: modelLoader.getMetrics(modelName),
          };
        } catch (err) {
          predictions[modelName] = {
            error: err.message,
          };
        }
      }

      res.json({
        success: true,
        modelCount: models.length,
        predictions,
      });
    }),
  );

  /* ============================================================
   * POST /api/predict/:modelName
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
        return res.status(400).json({
          success: false,
          error: validationError,
        });
      }

      try {
        const result = await modelLoader.predict(modelName, features);

        res.json({
          success: true,
          model: modelName,
          prediction: result.prediction,
          probability: result.probability,
          inputShape: result.inputShape,
          metrics: modelLoader.getMetrics(modelName),
        });
      } catch (err) {
        res.status(500).json({
          success: false,
          error: err.message,
        });
      }
    }),
  );

  return router;
}
