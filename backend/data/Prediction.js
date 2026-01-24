import mongoose from "mongoose";

const predictionSchema = new mongoose.Schema(
  {
    // Patient features used for prediction
    patientData: {
      age: { type: Number, required: true },
      sex: { type: Number, required: true }, // 1 = male, 0 = female
      cp: { type: Number, required: true }, // chest pain type (0-3)
      trestbps: { type: Number, required: true }, // resting blood pressure
      chol: { type: Number, required: true }, // serum cholesterol
      fbs: { type: Number, required: true }, // fasting blood sugar > 120 mg/dl
      restecg: { type: Number, required: true }, // resting ECG results (0-2)
      thalach: { type: Number, required: true }, // max heart rate achieved
      exang: { type: Number, required: true }, // exercise induced angina
      oldpeak: { type: Number, required: true }, // ST depression
      slope: { type: Number, required: true }, // slope of peak exercise ST segment
      ca: { type: Number, required: true }, // number of major vessels (0-3)
      thal: { type: Number, required: true }, // thalassemia (0-3)
    },

    // Prediction results
    prediction: {
      hasDisease: { type: Boolean, required: true },
      confidence: { type: Number, required: true }, // 0-1
      riskLevel: {
        type: String,
        enum: ["low", "medium", "high", "very-high"],
        required: true,
      },
    },

    // Model information
    modelInfo: {
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
      modelVersion: { type: String, required: true },
      accuracy: { type: Number }, // model accuracy at prediction time
      precision: { type: Number },
      recall: { type: Number },
      f1Score: { type: Number },
    },

    // Prediction probabilities (for all classes if multi-class)
    probabilities: {
      noDisease: { type: Number, required: true },
      hasDisease: { type: Number, required: true },
    },

    // Feature importance for this specific prediction (if available)
    featureImportance: [
      {
        feature: String,
        importance: Number,
      },
    ],

    // SHAP values or other explainability metrics (if available)
    explainability: {
      shapValues: [Number],
      topFeatures: [
        {
          feature: String,
          value: Number,
          impact: String, // 'positive' or 'negative'
        },
      ],
    },

    // Metadata
    metadata: {
      predictionTime: { type: Date, default: Date.now },
      processingTimeMs: { type: Number },
      userId: { type: String }, // if you have user authentication
      sessionId: { type: String },
      ipAddress: { type: String },
      userAgent: { type: String },
    },

    // Clinical notes or additional context
    notes: {
      doctorNotes: String,
      patientSymptoms: [String],
      medicalHistory: String,
    },

    // AI Explanation (generated after prediction)
    aiExplanation: {
      summary: String,
      riskFactors: [String],
      recommendations: [String],
      generatedAt: Date,
      modelUsed: String, // AI model used for explanation (e.g., 'gpt-4')
    },

    // Audit trail
    audit: {
      createdAt: { type: Date, default: Date.now },
      updatedAt: { type: Date, default: Date.now },
      isReviewed: { type: Boolean, default: false },
      reviewedBy: String,
      reviewedAt: Date,
      feedback: String,
      actualOutcome: Boolean, // for tracking actual patient outcomes
    },
  },
  {
    timestamps: true,
    collection: "predictions",
  },
);

// Indexes for efficient querying
predictionSchema.index({ "metadata.predictionTime": -1 });
predictionSchema.index({ "metadata.userId": 1 });
predictionSchema.index({ "prediction.hasDisease": 1 });
predictionSchema.index({ "prediction.riskLevel": 1 });
predictionSchema.index({ "modelInfo.modelName": 1 });
predictionSchema.index({ "metadata.sessionId": 1 });

// Virtual for risk percentage
predictionSchema.virtual("riskPercentage").get(function () {
  return Math.round(this.probabilities.hasDisease * 100);
});

// Method to get prediction summary
predictionSchema.methods.getSummary = function () {
  return {
    id: this._id,
    hasDisease: this.prediction.hasDisease,
    confidence: this.prediction.confidence,
    riskLevel: this.prediction.riskLevel,
    modelName: this.modelInfo.modelName,
    predictionTime: this.metadata.predictionTime,
  };
};

// Static method to get predictions by risk level
predictionSchema.statics.findByRiskLevel = function (riskLevel) {
  return this.find({ "prediction.riskLevel": riskLevel }).sort({
    "metadata.predictionTime": -1,
  });
};

// Static method to get recent predictions
predictionSchema.statics.getRecent = function (limit = 10) {
  return this.find().sort({ "metadata.predictionTime": -1 }).limit(limit);
};

// Pre-save middleware
predictionSchema.pre("save", function (next) {
  this.audit.updatedAt = new Date();
  next();
});

const Prediction =
  mongoose.models.Prediction || mongoose.model("Prediction", predictionSchema);

export default Prediction;
