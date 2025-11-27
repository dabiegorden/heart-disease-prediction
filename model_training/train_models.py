import os
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# ONNX exporters
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from data_preprocessing import CardiovascularDataPreprocessor

# ============================================================
#  CONFIG
# ============================================================
OUTPUT_DIR = "../backend/src/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_1 = "Cardiovascular_Disease_Dataset.xlsx"
DATASET_2 = "Heart_Disease x2.xlsx"

# ============================================================
#  LOAD + PREPROCESS BOTH DATASETS
# ============================================================
def load_full_dataset():
    print("📥 Loading datasets...")

    X1, y1, scaler = CardiovascularDataPreprocessor.load_and_prepare(
        DATASET_1, fit_scaler=True
    )

    X2, y2, _ = CardiovascularDataPreprocessor.load_and_prepare(
        DATASET_2, fit_scaler=False, scaler=scaler
    )

    print("📦 Combining datasets...")

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    print(f"→ Total samples: {len(y)}")
    return X, y

# ============================================================
#  TRAINING PIPELINE
# ============================================================
def train_models():
    X, y = load_full_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler.json
    scaler_json = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(f"{OUTPUT_DIR}/scaler.json", "w") as f:
        json.dump(scaler_json, f, indent=2)

    print("✓ scaler.json saved")

    # Models
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True),
        "logistic": LogisticRegression(max_iter=3000),
        "gradient_boost": GradientBoostingClassifier(),
    }

    metrics_out = {}

    # Train + export all models
    for name, model in models.items():
        print(f"\n🤖 Training: {name}")

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = (
            model.predict_proba(X_test_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Metrics
        metrics_out[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
            "cv_mean": float(cross_val_score(model, X_train_scaled, y_train, cv=5).mean()),
            "cv_std": float(cross_val_score(model, X_train_scaled, y_train, cv=5).std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # ONNX EXPORT
        try:
            print(f"→ Exporting {name}.onnx...")

            onnx_model = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
                options={
                    id(model): {
                        "zipmap": False,
                        "raw_scores": True,
                        "output_class_labels": False,
                    }
                }
            )

            with open(f"{OUTPUT_DIR}/{name}.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"✓ {name}.onnx saved")

        except Exception as e:
            print(f"⚠ ONNX export failed for {name}: {e}")
            joblib.dump(model, f"{OUTPUT_DIR}/{name}.pkl")
            print(f"✓ Saved fallback pickle model: {name}.pkl")

    # Save metrics
    with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n🎉 ALL MODELS TRAINED & EXPORTED SUCCESSFULLY!")

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    train_models()
