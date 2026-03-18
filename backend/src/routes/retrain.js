/**
 * routes/retrain.js
 * =================
 * Model retraining endpoints.
 *
 * Changes from original
 * ---------------------
 * 1. Training sessions are persisted to MongoDB (TrainingSession collection)
 *    instead of a flat JSON file.
 * 2. After a successful run the model's metrics are upserted in the
 *    ModelMetrics collection.
 * 3. File uploads now accept CSV **and** Excel (.xlsx / .xls) because the
 *    Python preprocessor handles both formats.
 * 4. The `multer` filename preserves the original extension so the Python
 *    script can detect the file format correctly.
 */

import express from "express";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";
import multer from "multer";
import { asyncHandler } from "../middleware/errorHandler.js";
import { executePythonScript } from "../utils/python-executor.js";
import { TrainingSession, ModelMetrics } from "../db/schema.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const apiRoot = path.resolve(__dirname, "../..");

const ALLOWED_EXTENSIONS = new Set([".csv", ".xlsx", ".xls"]);
const ALLOWED_MIMETYPES = new Set([
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);

/* ============================================================
 * MULTER – preserve original extension
 * ============================================================ */
const storage = multer.diskStorage({
  destination: async (_req, _file, cb) => {
    const uploadDir = path.join(apiRoot, "uploads");
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `dataset-${unique}${ext}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ALLOWED_EXTENSIONS.has(ext) || ALLOWED_MIMETYPES.has(file.mimetype)) {
      cb(null, true);
    } else {
      cb(
        new Error("Only CSV and Excel files (.csv, .xlsx, .xls) are accepted."),
      );
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
});

const VALID_MODELS = [
  "logistic_regression",
  "svm",
  "gradient_boost",
  "knn",
  "cnn1d",
  "cnn_lstm",
];

/* ============================================================
 * Helpers
 * ============================================================ */

/** Persist (or update) a TrainingSession document */
async function upsertSession(data) {
  try {
    await TrainingSession.findOneAndUpdate(
      { sessionId: data.sessionId },
      { $set: data },
      { upsert: true, new: true },
    );
  } catch (err) {
    console.error("[MongoDB] Failed to save session:", err.message);
  }
}

/** Upsert per-model metrics after successful training */
async function upsertMetrics(modelName, metrics, source = "retrained") {
  try {
    await ModelMetrics.findOneAndUpdate(
      { modelName },
      { $set: { ...metrics, modelName, source, trainedAt: new Date() } },
      { upsert: true, new: true },
    );
  } catch (err) {
    console.error("[MongoDB] Failed to save metrics:", err.message);
  }
}

/** Parse the TRAINING COMPLETE JSON block from Python stdout */
function parseTrainingResult(stdout) {
  const lines = stdout.split("\n");
  const completeIdx = lines.findIndex((l) =>
    l.includes("=== TRAINING COMPLETE ==="),
  );
  if (completeIdx === -1)
    throw new Error("Training completed but no results block found.");
  const jsonText = lines
    .slice(completeIdx + 1)
    .join("\n")
    .trim();
  return JSON.parse(jsonText);
}

/* ============================================================
 * ROUTER FACTORY
 * ============================================================ */
export default function createRetrainRouter() {
  const router = express.Router();

  /* ----------------------------------------------------------
   * POST /api/retrain/upload
   * Retrain one model with an uploaded dataset
   * ---------------------------------------------------------- */
  router.post(
    "/upload",
    upload.single("dataset"),
    asyncHandler(async (req, res) => {
      const { modelType, epochs = "50" } = req.body;

      if (!req.file) {
        return res
          .status(400)
          .json({ success: false, error: "No dataset file uploaded." });
      }
      if (!modelType) {
        return res
          .status(400)
          .json({ success: false, error: "modelType is required." });
      }
      if (!VALID_MODELS.includes(modelType)) {
        return res.status(400).json({
          success: false,
          error: `Invalid modelType. Must be one of: ${VALID_MODELS.join(", ")}`,
        });
      }

      const sessionId = Date.now().toString();
      const dataPath = req.file.path;
      const outputDir = path.join(process.cwd(), "models", "retrained");

      const session = {
        sessionId,
        modelType,
        status: "training",
        startTime: new Date(),
        originalFilename: req.file.originalname,
        fileSizeBytes: req.file.size,
        progress: 10,
        currentMessage: "Initializing…",
        metrics: null,
        error: null,
      };

      await upsertSession(session);

      // Respond immediately so the client can poll /status/:sessionId
      res.json({
        success: true,
        sessionId,
        message: `Training ${modelType} started`,
        modelType,
      });

      // ── Background training ────────────────────────────────
      try {
        const result = await executePythonScript(
          "model_retrainer.py",
          [
            "--model-type",
            modelType,
            "--data-path",
            dataPath,
            "--output-dir",
            outputDir,
            "--epochs",
            epochs,
          ],
          {
            onProgress: async (pct, msg) => {
              await upsertSession({
                sessionId,
                progress: Math.max(session.progress, pct),
                currentMessage: msg || "Training…",
              });
            },
          },
        );

        const trainingResult = parseTrainingResult(result.stdout);

        const completed = {
          sessionId,
          status: "completed",
          endTime: new Date(),
          metrics: trainingResult.metrics,
          numSamples: trainingResult.num_samples,
          numFeatures: trainingResult.num_features,
          featureNames: trainingResult.feature_names,
          progress: 100,
          currentMessage: "Training complete!",
        };
        await upsertSession(completed);
        await upsertMetrics(modelType, trainingResult.metrics);
      } catch (err) {
        console.error(`[Retrain] ${modelType} failed:`, err.message);
        await upsertSession({
          sessionId,
          status: "failed",
          endTime: new Date(),
          error: err.message,
          progress: 0,
          currentMessage: `Failed: ${err.message}`,
        });
      } finally {
        fs.unlink(dataPath).catch(() => {});
      }
    }),
  );

  /* ----------------------------------------------------------
   * POST /api/retrain/train-all
   * Retrain all 6 models sequentially with one uploaded dataset
   * ---------------------------------------------------------- */
  router.post(
    "/train-all",
    upload.single("dataset"),
    asyncHandler(async (req, res) => {
      const { epochs = "50" } = req.body;

      if (!req.file) {
        return res
          .status(400)
          .json({ success: false, error: "No dataset file uploaded." });
      }

      const sessionId = Date.now().toString();
      const dataPath = req.file.path;
      const outputDir = path.join(process.cwd(), "models", "retrained");

      const session = {
        sessionId,
        modelType: "all",
        status: "training",
        startTime: new Date(),
        originalFilename: req.file.originalname,
        fileSizeBytes: req.file.size,
        progress: 0,
        currentMessage: "Starting…",
        currentModel: null,
        results: {},
        error: null,
      };

      await upsertSession(session);

      res.json({
        success: true,
        sessionId,
        message: "Training all 6 models started",
        totalModels: VALID_MODELS.length,
      });

      // ── Sequential training loop ───────────────────────────
      let completed = 0;

      for (const modelType of VALID_MODELS) {
        await upsertSession({
          sessionId,
          currentModel: modelType,
          currentMessage: `Training ${modelType}…`,
        });

        try {
          const result = await executePythonScript(
            "model_retrainer.py",
            [
              "--model-type",
              modelType,
              "--data-path",
              dataPath,
              "--output-dir",
              outputDir,
              "--epochs",
              epochs,
            ],
            {
              onProgress: async (pct, msg) => {
                const base = (completed / VALID_MODELS.length) * 100;
                await upsertSession({
                  sessionId,
                  progress: Math.round(base + pct / VALID_MODELS.length),
                  currentMessage: `${modelType}: ${msg}`,
                });
              },
            },
          );

          const trainingResult = parseTrainingResult(result.stdout);
          session.results[modelType] = trainingResult.metrics;
          await upsertMetrics(modelType, trainingResult.metrics);
        } catch (err) {
          console.error(`[Retrain-All] ${modelType} failed:`, err.message);
          session.results[modelType] = { error: err.message };
        }

        completed++;
        await upsertSession({
          sessionId,
          results: { ...session.results },
          progress: Math.round((completed / VALID_MODELS.length) * 100),
          currentMessage: `Completed ${modelType}`,
        });
      }

      await upsertSession({
        sessionId,
        status: "completed",
        endTime: new Date(),
        currentMessage: "All models trained!",
        results: session.results,
      });

      fs.unlink(dataPath).catch(() => {});
      console.log("[Retrain-All] Done");
    }),
  );

  /* ----------------------------------------------------------
   * GET /api/retrain/status/:sessionId
   * ---------------------------------------------------------- */
  router.get(
    "/status/:sessionId",
    asyncHandler(async (req, res) => {
      const session = await TrainingSession.findOne({
        sessionId: req.params.sessionId,
      }).lean();

      if (!session) {
        return res
          .status(404)
          .json({ success: false, error: "Session not found." });
      }

      res.json({ success: true, session });
    }),
  );

  /* ----------------------------------------------------------
   * GET /api/retrain/results
   * All completed sessions
   * ---------------------------------------------------------- */
  router.get(
    "/results",
    asyncHandler(async (req, res) => {
      const limit = Math.min(parseInt(req.query.limit ?? "50", 10), 200);
      const sessions = await TrainingSession.find({ status: "completed" })
        .sort({ endTime: -1 })
        .limit(limit)
        .lean();

      res.json({
        success: true,
        total: sessions.length,
        sessions,
      });
    }),
  );

  /* ----------------------------------------------------------
   * DELETE /api/retrain/sessions/:sessionId
   * ---------------------------------------------------------- */
  router.delete(
    "/sessions/:sessionId",
    asyncHandler(async (req, res) => {
      const result = await TrainingSession.deleteOne({
        sessionId: req.params.sessionId,
      });

      if (result.deletedCount === 0) {
        return res
          .status(404)
          .json({ success: false, error: "Session not found." });
      }

      res.json({ success: true, message: "Session deleted." });
    }),
  );

  return router;
}
