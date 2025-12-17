import os
import json
import joblib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from deep_learning_models import CNN1D, CNNLSTM

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

DEVICE = torch.device("cpu")
EPOCHS = 30
BATCH_SIZE = 32

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
#  TRAIN DEEP LEARNING MODELS
# ============================================================

    model.to(DEVICE)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1),
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model


def evaluate_dl_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()

    y_pred = (preds > 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

def train_dl_model(model, X_train, y_train):
    model.to(DEVICE)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model

# ============================================================
#  TRAINING PIPELINE
# ============================================================
def train_models():
    X, y = load_full_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(f"{OUTPUT_DIR}/scaler.json", "w") as f:
        json.dump(
            {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            f,
            indent=2,
        )

    print("✓ scaler.json saved")

    metrics_out = {}

    # --------------------------------------------------------
    # CLASSICAL ML MODELS
    # --------------------------------------------------------
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True),
        "logistic": LogisticRegression(max_iter=3000),
        "gradient_boost": GradientBoostingClassifier(),
    }

    for name, model in models.items():
        print(f"\n🤖 Training: {name}")

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics_out[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_prob)),
            "cv_mean": float(cross_val_score(model, X_train_scaled, y_train, cv=5).mean()),
            "cv_std": float(cross_val_score(model, X_train_scaled, y_train, cv=5).std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"→ Exporting {name}.onnx")

        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {"zipmap": False}},
        )

        with open(f"{OUTPUT_DIR}/{name}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✓ {name}.onnx saved")

    # --------------------------------------------------------
    # DEEP LEARNING MODELS
    # --------------------------------------------------------
    print("\n🧠 Training Deep Learning Models")

    num_features = X_train_scaled.shape[1]
    X_train_dl = X_train_scaled.reshape(-1, 1, num_features)
    X_test_dl = X_test_scaled.reshape(-1, 1, num_features)

    # ---------------- CNN ----------------
    cnn = CNN1D(num_features)
    cnn = train_dl_model(cnn, X_train_dl, y_train)

    metrics_out["cnn"] = evaluate_dl_model(cnn, X_test_dl, y_test)

    torch.onnx.export(
        cnn,
        torch.randn(1, 1, num_features),
        f"{OUTPUT_DIR}/cnn.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
)


    print("✓ cnn.onnx saved")

    # ---------------- CNN + LSTM ----------------
    cnn_lstm = CNNLSTM(num_features)
    cnn_lstm = train_dl_model(cnn_lstm, X_train_dl, y_train)

    metrics_out["cnn_lstm"] = evaluate_dl_model(cnn_lstm, X_test_dl, y_test)

    torch.onnx.export(
        cnn_lstm,
        torch.randn(1, 1, num_features),
        f"{OUTPUT_DIR}/cnn_lstm.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
)


    print("✓ cnn_lstm.onnx saved")

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------
    with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n🎉 ALL MODELS (ML + DL) TRAINED & EXPORTED SUCCESSFULLY!")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    train_models()
