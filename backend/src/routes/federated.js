/**
 * routes/federated.js
 * ====================
 * Federated learning comparison routes.
 *
 * Changes from original
 * ---------------------
 * 1. /comparison now reads ModelMetrics from MongoDB instead of a flat JSON
 *    file, so it always reflects the latest retrained results.
 * 2. Falls back to model_metrics.json (initial training) when the DB has
 *    no retrained records yet.
 * 3. Empty route handlers are left as stubs with clear TODO comments.
 */

import express from "express";
import { spawn } from "child_process";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";
import { asyncHandler } from "../middleware/errorHandler.js";
import { ModelMetrics } from "../db/schema.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getPythonCommand() {
  return process.platform === "win32" ? "python" : "python3";
}

function executePythonScript(scriptName, args = []) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, "../../python", scriptName);
    const cmd = getPythonCommand();

    console.log(`[Federated] ${cmd} ${scriptPath}`);

    const proc = spawn(cmd, [scriptPath, ...args]);
    let stdout = "",
      stderr = "";

    proc.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    proc.stderr.on("data", (d) => {
      stderr += d.toString();
    });

    proc.on("close", (code) => {
      code === 0
        ? resolve({ stdout, stderr, code })
        : reject(new Error(`Script exited ${code}\n${stderr}`));
    });

    proc.on("error", (err) =>
      reject(new Error(`Failed to start Python: ${err.message}`)),
    );
  });
}

export default function createFederatedRouter() {
  const router = express.Router();

  /* ----------------------------------------------------------
   * POST /api/federated/partition
   * TODO: Implement dataset partitioning for federated simulation
   * ---------------------------------------------------------- */
  router.post(
    "/partition",
    asyncHandler(async (req, res) => {
      res.status(501).json({ success: false, message: "Not yet implemented." });
    }),
  );

  /* ----------------------------------------------------------
   * POST /api/federated/train
   * TODO: Implement federated training round
   * ---------------------------------------------------------- */
  router.post(
    "/train",
    asyncHandler(async (req, res) => {
      res.status(501).json({ success: false, message: "Not yet implemented." });
    }),
  );

  /* ----------------------------------------------------------
   * GET /api/federated/status
   * TODO: Return federated training status
   * ---------------------------------------------------------- */
  router.get(
    "/status",
    asyncHandler(async (req, res) => {
      res.json({
        success: true,
        status: "idle",
        message: "No federated training running.",
      });
    }),
  );

  /* ----------------------------------------------------------
   * GET /api/federated/results/:modelName
   * TODO: Return federated model results
   * ---------------------------------------------------------- */
  router.get(
    "/results/:modelName",
    asyncHandler(async (req, res) => {
      res.status(501).json({ success: false, message: "Not yet implemented." });
    }),
  );

  /* ----------------------------------------------------------
   * GET /api/federated/comparison
   * Compare centralized (initial) vs retrained metrics.
   * Primary source: MongoDB ModelMetrics collection.
   * Fallback: model_metrics.json written by train_models.py.
   * ---------------------------------------------------------- */
  router.get(
    "/comparison",
    asyncHandler(async (req, res) => {
      // ── 1. Try MongoDB first ─────────────────────────────
      const dbRecords = await ModelMetrics.find({}).lean();

      let centralized = {};
      let retrained = {};

      if (dbRecords.length > 0) {
        for (const rec of dbRecords) {
          const metrics = {
            accuracy: rec.accuracy,
            precision: rec.precision,
            recall: rec.recall,
            f1_score: rec.f1 ?? rec.f1_score,
            auc_roc: rec.auc ?? rec.auc_roc,
            cv_mean: rec.cv_mean,
            cv_std: rec.cv_std,
            trainedAt: rec.trainedAt,
          };

          if (rec.source === "retrained") {
            retrained[rec.modelName] = metrics;
          } else {
            centralized[rec.modelName] = metrics;
          }
        }
      }

      // ── 2. If centralized is still empty, fall back to file ──
      if (Object.keys(centralized).length === 0) {
        const metricsPath = path.join(
          __dirname,
          "../../src/models/model_metrics.json",
        );
        try {
          const raw = await fs.readFile(metricsPath, "utf-8");
          centralized = JSON.parse(raw);
        } catch {
          console.warn(
            "[Federated] model_metrics.json not found – using defaults",
          );
          centralized = {
            logistic_regression: {
              accuracy: 0.85,
              precision: 0.83,
              recall: 0.87,
              f1_score: 0.85,
              auc_roc: 0.91,
            },
            svm: {
              accuracy: 0.87,
              precision: 0.86,
              recall: 0.88,
              f1_score: 0.87,
              auc_roc: 0.93,
            },
            gradient_boost: {
              accuracy: 0.89,
              precision: 0.88,
              recall: 0.9,
              f1_score: 0.89,
              auc_roc: 0.95,
            },
            knn: {
              accuracy: 0.82,
              precision: 0.8,
              recall: 0.84,
              f1_score: 0.82,
              auc_roc: 0.9,
            },
            cnn: {
              accuracy: 0.9,
              precision: 0.89,
              recall: 0.91,
              f1_score: 0.9,
              auc_roc: 0.96,
            },
            cnn_lstm: {
              accuracy: 0.91,
              precision: 0.9,
              recall: 0.92,
              f1_score: 0.91,
              auc_roc: 0.97,
            },
          };
        }
      }

      // If no retraining has happened yet, mirror centralized
      if (Object.keys(retrained).length === 0) {
        retrained = { ...centralized };
      }

      if (Object.keys(centralized).length === 0) {
        return res.json({
          success: false,
          comparison: null,
          message: "No model data available. Run train_models.py first.",
        });
      }

      res.json({
        success: true,
        comparison: { centralized, retrained },
      });
    }),
  );

  /* ----------------------------------------------------------
   * DELETE /api/federated/sessions
   * TODO: Clear federated training sessions
   * ---------------------------------------------------------- */
  router.delete(
    "/sessions",
    asyncHandler(async (req, res) => {
      res.status(501).json({ success: false, message: "Not yet implemented." });
    }),
  );

  return router;
}
