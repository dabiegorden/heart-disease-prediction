/**
 * Model Retraining Routes
 * Simplified API for clients to retrain models with their own datasets
 */

import express from "express";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";
import multer from "multer";
import { asyncHandler } from "../middleware/errorHandler.js";
import { executePythonScript } from "../utils/python-executor.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const apiRoot = path.resolve(__dirname, "../..");

const SESSIONS_FILE = path.join(apiRoot, "data", "training-sessions.json");

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(apiRoot, "uploads");
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, `dataset-${uniqueSuffix}.csv`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, true);
    } else {
      cb(new Error("Only CSV files are allowed"));
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

async function loadSessions() {
  try {
    const dataDir = path.dirname(SESSIONS_FILE);
    await fs.mkdir(dataDir, { recursive: true });

    const data = await fs.readFile(SESSIONS_FILE, "utf-8");
    const sessions = JSON.parse(data);
    return new Map(Object.entries(sessions));
  } catch (error) {
    // File doesn't exist or is invalid, return empty Map
    return new Map();
  }
}

async function saveSessions(trainingSessions) {
  try {
    const dataDir = path.dirname(SESSIONS_FILE);
    await fs.mkdir(dataDir, { recursive: true });

    const sessions = Object.fromEntries(trainingSessions);
    await fs.writeFile(SESSIONS_FILE, JSON.stringify(sessions, null, 2));
  } catch (error) {
    console.error("[v0] Failed to save sessions:", error);
  }
}

export default function createRetrainRouter() {
  const router = express.Router();

  let trainingSessions = new Map();

  loadSessions().then((sessions) => {
    trainingSessions = sessions;
    console.log(`[v0] Loaded ${sessions.size} training sessions from disk`);
  });

  /* ============================================================
   * POST /api/retrain/upload
   * Upload dataset and retrain a specific model
   * ============================================================ */
  router.post(
    "/upload",
    upload.single("dataset"),
    asyncHandler(async (req, res) => {
      const { modelType, epochs = "50" } = req.body;

      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No dataset file uploaded",
        });
      }

      if (!modelType) {
        return res.status(400).json({
          success: false,
          error: "Model type is required",
        });
      }

      const validModels = [
        "logistic_regression",
        "svm",
        "gradient_boost",
        "knn",
        "cnn1d",
        "cnn_lstm",
      ];

      if (!validModels.includes(modelType)) {
        return res.status(400).json({
          success: false,
          error: `Invalid model type. Must be one of: ${validModels.join(
            ", ",
          )}`,
        });
      }

      console.log(`[Retrain] Received dataset: ${req.file.filename}`);
      console.log(`[Retrain] Model type: ${modelType}`);
      console.log(`[Retrain] File size: ${req.file.size} bytes`);

      const sessionId = Date.now().toString();
      const dataPath = req.file.path;
      const outputDir = path.join(process.cwd(), "models", "retrained");

      const session = {
        sessionId,
        modelType,
        status: "training",
        startTime: new Date().toISOString(),
        dataPath,
        progress: 10,
        currentMessage: "Initializing...",
        metrics: null,
        error: null,
      };

      trainingSessions.set(sessionId, session);
      await saveSessions(trainingSessions);

      res.json({
        success: true,
        sessionId,
        message: `Training ${modelType} model started`,
        modelType,
      });

      try {
        console.log(`[v0] Starting training for ${modelType}`);
        console.log(`[v0] Data path: ${dataPath}`);
        console.log(`[v0] Output dir: ${outputDir}`);

        const args = [
          "--model-type",
          modelType,
          "--data-path",
          dataPath,
          "--output-dir",
          outputDir,
          "--epochs",
          epochs,
        ];

        const result = await executePythonScript("model_retrainer.py", args, {
          onProgress: async (percentage, message) => {
            console.log(`[v0] Training progress: ${percentage}% - ${message}`);
            session.progress = Math.max(session.progress, percentage);
            session.currentMessage = message || "Training...";
            await saveSessions(trainingSessions);
          },
        });

        console.log(`[v0] Training output:`, result.stdout);

        const lines = result.stdout.split("\n");
        const completeIdx = lines.findIndex((line) =>
          line.includes("=== TRAINING COMPLETE ==="),
        );

        if (completeIdx !== -1) {
          const jsonLines = lines
            .slice(completeIdx + 1)
            .join("\n")
            .trim();
          const trainingResult = JSON.parse(jsonLines);

          session.status = "completed";
          session.endTime = new Date().toISOString();
          session.metrics = trainingResult.metrics;
          session.progress = 100;
          session.currentMessage = "Training completed!";

          console.log(
            `[v0] Training completed for ${modelType}:`,
            trainingResult.metrics,
          );
        } else {
          throw new Error("Training completed but no results found");
        }
      } catch (error) {
        console.error(`[v0] Training failed for ${modelType}:`, error);
        session.status = "failed";
        session.error = error.message;
        session.endTime = new Date().toISOString();
        session.progress = 0;
        session.currentMessage = `Failed: ${error.message}`;
      } finally {
        await saveSessions(trainingSessions);

        try {
          await fs.unlink(dataPath);
        } catch (err) {
          console.warn(`[v0] Failed to delete uploaded file: ${dataPath}`);
        }
      }
    }),
  );

  /* ============================================================
   * POST /api/retrain/train-all
   * Train all 6 models with uploaded dataset
   * ============================================================ */
  router.post(
    "/train-all",
    upload.single("dataset"),
    asyncHandler(async (req, res) => {
      const { epochs = "50" } = req.body;

      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No dataset file uploaded",
        });
      }

      console.log(`[Retrain All] Received dataset: ${req.file.filename}`);

      const sessionId = Date.now().toString();
      const dataPath = req.file.path;
      const outputDir = path.join(process.cwd(), "models", "retrained");
      const models = [
        "logistic_regression",
        "svm",
        "gradient_boost",
        "knn",
        "cnn1d",
        "cnn_lstm",
      ];

      const session = {
        sessionId,
        modelType: "all",
        status: "training",
        startTime: new Date().toISOString(),
        dataPath,
        progress: 0,
        currentMessage: "Starting...",
        currentModel: null,
        results: {},
        error: null,
      };

      trainingSessions.set(sessionId, session);
      await saveSessions(trainingSessions);

      res.json({
        success: true,
        sessionId,
        message: "Training all 6 models started",
        totalModels: models.length,
      });

      let completedCount = 0;

      for (const modelType of models) {
        session.currentModel = modelType;
        session.currentMessage = `Training ${modelType}...`;
        await saveSessions(trainingSessions);

        try {
          console.log(
            `[v0] Training ${modelType} (${completedCount + 1}/${
              models.length
            })`,
          );

          const args = [
            "--model-type",
            modelType,
            "--data-path",
            dataPath,
            "--output-dir",
            outputDir,
            "--epochs",
            epochs,
          ];

          const result = await executePythonScript("model_retrainer.py", args, {
            onProgress: async (percentage, message) => {
              const baseProgress = (completedCount / models.length) * 100;
              const modelProgress = percentage / models.length;
              session.progress = Math.round(baseProgress + modelProgress);
              session.currentMessage = `${modelType}: ${message}`;
              await saveSessions(trainingSessions);
            },
          });

          const lines = result.stdout.split("\n");
          const completeIdx = lines.findIndex((line) =>
            line.includes("=== TRAINING COMPLETE ==="),
          );

          if (completeIdx !== -1) {
            const jsonLines = lines
              .slice(completeIdx + 1)
              .join("\n")
              .trim();
            const trainingResult = JSON.parse(jsonLines);
            session.results[modelType] = trainingResult.metrics;

            console.log(`[v0] Completed ${modelType}:`, trainingResult.metrics);
          } else {
            session.results[modelType] = { error: "No results found" };
          }

          completedCount++;
          session.progress = Math.round((completedCount / models.length) * 100);
          session.currentMessage = `Completed ${modelType}`;
          await saveSessions(trainingSessions);
        } catch (error) {
          console.error(`[v0] Failed ${modelType}:`, error.message);
          session.results[modelType] = { error: error.message };
          completedCount++;
          session.progress = Math.round((completedCount / models.length) * 100);
          session.currentMessage = `Failed ${modelType}: ${error.message}`;
          await saveSessions(trainingSessions);
        }
      }

      session.status = "completed";
      session.endTime = new Date().toISOString();
      session.currentMessage = "All models trained!";
      await saveSessions(trainingSessions);

      try {
        await fs.unlink(dataPath);
      } catch (err) {
        console.warn(`[v0] Failed to delete uploaded file: ${dataPath}`);
      }

      console.log(`[v0] All models training completed`);
    }),
  );

  /* ============================================================
   * GET /api/retrain/status/:sessionId
   * Get training status
   * ============================================================ */
  router.get(
    "/status/:sessionId",
    asyncHandler(async (req, res) => {
      const { sessionId } = req.params;

      const session = trainingSessions.get(sessionId);

      if (!session) {
        return res.status(404).json({
          success: false,
          error: "Training session not found",
        });
      }

      res.json({
        success: true,
        session,
      });
    }),
  );

  /* ============================================================
   * GET /api/retrain/results
   * Get all training results
   * ============================================================ */
  router.get(
    "/results",
    asyncHandler(async (req, res) => {
      const sessions = Array.from(trainingSessions.values());
      const completed = sessions.filter((s) => s.status === "completed");

      console.log(`[v0] Returning ${completed.length} completed sessions`);

      res.json({
        success: true,
        total: sessions.length,
        completed: completed.length,
        sessions: completed,
      });
    }),
  );

  /* ============================================================
   * DELETE /api/retrain/sessions/:sessionId
   * Delete a training session
   * ============================================================ */
  router.delete(
    "/sessions/:sessionId",
    asyncHandler(async (req, res) => {
      const { sessionId } = req.params;

      if (trainingSessions.has(sessionId)) {
        trainingSessions.delete(sessionId);
        await saveSessions(trainingSessions);
        res.json({
          success: true,
          message: "Session deleted",
        });
      } else {
        res.status(404).json({
          success: false,
          error: "Session not found",
        });
      }
    }),
  );

  return router;
}
