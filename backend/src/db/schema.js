/**
 * MongoDB Schemas
 * ===============
 * Three collections:
 *   1. predictions      – every inference request + result
 *   2. trainingSessions – retrain jobs (upload → train → complete/fail)
 *   3. modelMetrics     – latest evaluation metrics per model
 */

import mongoose from "mongoose";

const { Schema, model } = mongoose;

/* ============================================================
 * 1. PREDICTION
 * ============================================================ */
const predictionSchema = new Schema(
  {
    // Which model(s) were used
    modelName: { type: String, required: true, index: true },

    // Raw input features as submitted by the client
    inputFeatures: {
      type: Map,
      of: Number,
      required: true,
    },

    // Scaled feature vector actually fed to the model
    scaledFeatures: [Number],

    // Output
    prediction: { type: Number, enum: [0, 1], required: true },
    probability: { type: Number, min: 0, max: 1, required: true },
    label: {
      type: String,
      enum: ["Heart Disease Likely", "Heart Disease Unlikely"],
    },

    // Optional – compare endpoint stores all model results here
    allModelResults: { type: Schema.Types.Mixed, default: null },

    // Request metadata
    requestIp: String,
    userAgent: String,
  },
  {
    timestamps: true, // createdAt, updatedAt
    collection: "predictions",
  },
);

/* ============================================================
 * 2. TRAINING SESSION
 * ============================================================ */
const trainingSessionSchema = new Schema(
  {
    sessionId: { type: String, required: true, unique: true, index: true },
    modelType: { type: String, required: true }, // "knn" | "all" | …
    status: {
      type: String,
      enum: ["pending", "training", "completed", "failed"],
      default: "pending",
      index: true,
    },

    // Dataset info
    originalFilename: String,
    fileSizeBytes: Number,
    numSamples: Number,
    numFeatures: Number,
    featureNames: [String],

    // Progress tracking
    progress: { type: Number, default: 0, min: 0, max: 100 },
    currentMessage: { type: String, default: "" },
    currentModel: String, // used when modelType === "all"

    // Results
    metrics: { type: Schema.Types.Mixed, default: null }, // single model
    results: { type: Schema.Types.Mixed, default: {} }, // all-models map

    // Timing
    startTime: { type: Date, default: Date.now },
    endTime: Date,

    // Error info
    error: String,
  },
  {
    timestamps: true,
    collection: "trainingSessions",
  },
);

/* ============================================================
 * 3. MODEL METRICS  (upserted after every training run)
 * ============================================================ */
const modelMetricsSchema = new Schema(
  {
    modelName: { type: String, required: true, unique: true, index: true },
    source: {
      type: String,
      enum: ["initial", "retrained"],
      default: "initial",
    },

    accuracy: Number,
    precision: Number,
    recall: Number,
    f1: Number,
    f1_score: Number, // retrain.py uses f1_score key
    auc: Number,
    auc_roc: Number, // retrain.py uses auc_roc key
    cv_mean: Number,
    cv_std: Number,
    confusion_matrix: [[Number]],

    trainedAt: { type: Date, default: Date.now },
  },
  {
    timestamps: true,
    collection: "modelMetrics",
  },
);

export const Prediction = model("Prediction", predictionSchema);
export const TrainingSession = model("TrainingSession", trainingSessionSchema);
export const ModelMetrics = model("ModelMetrics", modelMetricsSchema);
