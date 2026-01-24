import mongoose from "mongoose";

const trainedModelSchema = new mongoose.Schema(
  {
    // Model identification
    modelName: {
      type: String,
      required: true,
      enum: [
        "random-forest",
        "gradient-boosting",
        "svm",
        "neural-network",
        "knn",
        "logistic-regression",
        "ensemble",
      ],
    },
    version: { type: String, required: true },
    description: String,

    // Training information
    training: {
      trainingDate: { type: Date, required: true },
      datasetSize: { type: Number, required: true },
      trainTestSplit: {
        trainSize: Number,
        testSize: Number,
      },
      features: [String], // feature names used
      targetVariable: { type: String, default: "target" },

      // Hyperparameters used during training
      hyperparameters: mongoose.Schema.Types.Mixed,

      // Training duration
      trainingTimeSeconds: Number,

      // Dataset info
      datasetInfo: {
        name: String,
        version: String,
        source: String,
        preprocessingSteps: [String],
      },
    },

    // Performance metrics
    performance: {
      // Overall metrics
      accuracy: { type: Number, required: true },
      precision: { type: Number, required: true },
      recall: { type: Number, required: true },
      f1Score: { type: Number, required: true },

      // Additional metrics
      auc: Number, // Area Under the ROC Curve
      rocCurve: [
        {
          fpr: Number,
          tpr: Number,
          threshold: Number,
        },
      ],

      // Confusion matrix
      confusionMatrix: {
        truePositive: Number,
        trueNegative: Number,
        falsePositive: Number,
        falseNegative: Number,
      },

      // Cross-validation scores
      crossValidation: {
        scores: [Number],
        mean: Number,
        std: Number,
        folds: Number,
      },

      // Class-specific metrics (if multi-class)
      classMetrics: [
        {
          class: String,
          precision: Number,
          recall: Number,
          f1Score: Number,
          support: Number,
        },
      ],
    },

    // Feature importance
    featureImportance: [
      {
        feature: { type: String, required: true },
        importance: { type: Number, required: true },
        rank: Number,
      },
    ],

    // Model file information
    modelFile: {
      format: {
        type: String,
        enum: ["onnx", "pickle", "h5", "savedmodel", "joblib"],
        default: "onnx",
      },
      path: String, // file path or URL
      size: Number, // file size in bytes
      checksum: String, // MD5 or SHA256 hash
      uploadedAt: Date,
    },

    // Deployment status
    deployment: {
      status: {
        type: String,
        enum: ["development", "staging", "production", "archived"],
        default: "development",
      },
      deployedAt: Date,
      endpoint: String,
      isActive: { type: Boolean, default: false },

      // Usage stats
      usageStats: {
        totalPredictions: { type: Number, default: 0 },
        avgResponseTime: Number, // milliseconds
        lastUsed: Date,
        errorRate: Number, // percentage
      },
    },

    // Comparison with previous versions
    comparison: {
      previousVersion: String,
      improvementMetrics: {
        accuracyDelta: Number,
        precisionDelta: Number,
        recallDelta: Number,
        f1ScoreDelta: Number,
      },
      notes: String,
    },

    // Metadata
    metadata: {
      createdBy: String,
      framework: {
        type: String,
        enum: [
          "scikit-learn",
          "tensorflow",
          "pytorch",
          "xgboost",
          "lightgbm",
          "keras",
        ],
        default: "scikit-learn",
      },
      pythonVersion: String,
      frameworkVersion: String,
      dependencies: [String],

      // Tags for organization
      tags: [String],

      // Notes and documentation
      notes: String,
      trainingNotes: String,
      knownIssues: [String],
      changelog: [String],
    },

    // Validation and testing
    validation: {
      validationSet: {
        size: Number,
        accuracy: Number,
        precision: Number,
        recall: Number,
      },
      testSet: {
        size: Number,
        accuracy: Number,
        precision: Number,
        recall: Number,
      },
    },

    // Audit trail
    audit: {
      createdAt: { type: Date, default: Date.now },
      updatedAt: { type: Date, default: Date.now },
      lastEvaluatedAt: Date,
      evaluationCount: { type: Number, default: 0 },
    },
  },
  {
    timestamps: true,
    collection: "trained_models",
  },
);

// Indexes
trainedModelSchema.index({ modelName: 1, version: 1 }, { unique: true });
trainedModelSchema.index({ "deployment.status": 1 });
trainedModelSchema.index({ "deployment.isActive": 1 });
trainedModelSchema.index({ "performance.accuracy": -1 });
trainedModelSchema.index({ "training.trainingDate": -1 });

// Virtual for formatted accuracy
trainedModelSchema.virtual("accuracyPercentage").get(function () {
  return `${(this.performance.accuracy * 100).toFixed(2)}%`;
});

// Method to get performance summary
trainedModelSchema.methods.getPerformanceSummary = function () {
  return {
    modelName: this.modelName,
    version: this.version,
    accuracy: this.performance.accuracy,
    precision: this.performance.precision,
    recall: this.performance.recall,
    f1Score: this.performance.f1Score,
    totalPredictions: this.deployment.usageStats.totalPredictions,
  };
};

// Method to increment prediction count
trainedModelSchema.methods.incrementPredictionCount = async function () {
  this.deployment.usageStats.totalPredictions += 1;
  this.deployment.usageStats.lastUsed = new Date();
  await this.save();
};

// Static method to get active production models
trainedModelSchema.statics.getActiveModels = function () {
  return this.find({
    "deployment.status": "production",
    "deployment.isActive": true,
  }).sort({ "performance.accuracy": -1 });
};

// Static method to get best performing model by name
trainedModelSchema.statics.getBestModel = function (modelName) {
  return this.findOne({ modelName })
    .sort({ "performance.accuracy": -1, "training.trainingDate": -1 })
    .limit(1);
};

// Pre-save middleware
trainedModelSchema.pre("save", function (next) {
  this.audit.updatedAt = new Date();
  next();
});

const TrainedModel =
  mongoose.models.TrainedModel ||
  mongoose.model("TrainedModel", trainedModelSchema);

export default TrainedModel;
