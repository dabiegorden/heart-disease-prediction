"""
train_models.py
===============
Train classical ML and deep learning models on any combination of
heart / cardiovascular disease datasets, then export every model as ONNX
so the Node.js / Express backend can serve predictions.

Key improvements over the original script
------------------------------------------
1. Uses the universal ``CardiovascularDataPreprocessor`` – no hard-coded
   column names.  Any file whose columns can be aliased (or are new) is
   accepted automatically.
2. The ONNX ``input_size`` is derived from the *actual* feature count so
   the exported graph always matches the data.
3. Deep-learning models use ``AdaptiveAvgPool1d`` / LSTM so they accept any
   feature width without architecture changes.
4. ``scaler.json`` also records the canonical ``feature_names`` list so the
   backend knows exactly what to expect.
5. Duplicate ``train_dl_model`` function removed; leftover dead code cleaned.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from data_preprocessing import CardiovascularDataPreprocessor
from deep_learning_models import CNN1D, CNNLSTM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("../backend/src/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── List every dataset file you want to train on ──────────────────────────
# The preprocessor handles different column sets automatically.
# Add or remove paths here; no other code needs to change.
DATASET_FILES: list[str] = [
    "Cardiovascular_Disease_Dataset.xlsx",
    "Heart_Disease x2.xlsx",
    "Heart_Disease x3.xlsx",
]

DEVICE     = torch.device("cpu")
EPOCHS     = 30
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_full_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and merge all configured datasets into a single (X, y) pair.

    Path resolution
    ---------------
    Datasets are resolved relative to this script's own directory so the
    script works correctly no matter which working directory it is launched
    from (e.g. ``python model_training/train_models.py`` from the project
    root still finds the files).

    Missing columns in any dataset are automatically filled with the
    training-set column mean and a warning is logged – no manual mapping
    is required when a new hospital dataset has a different column set.
    """
    # Always resolve paths relative to THIS file, not the cwd
    base = Path(__file__).resolve().parent

    log.info("Script directory : %s", base)
    log.info("Looking for datasets: %s", DATASET_FILES)

    # ── Verify every file exists before doing any work ─────────────────
    paths: list[Path] = []
    missing: list[str] = []

    for filename in DATASET_FILES:
        full_path = base / filename
        if full_path.exists():
            paths.append(full_path)
            log.info("  ✓  found  →  %s", full_path.name)
        else:
            missing.append(str(full_path))
            log.error("  ✗  NOT FOUND  →  %s", full_path)

    if missing:
        raise FileNotFoundError(
            f"\n\n{len(missing)} dataset file(s) could not be found:\n"
            + "\n".join(f"  • {p}" for p in missing)
            + f"\n\nExpected location: {base}\n"
            + "Make sure the filenames in DATASET_FILES match exactly "
            "(including spaces, capitalisation, and extension)."
        )

    # ── Load, align columns, scale ─────────────────────────────────────
    X, y, prep = CardiovascularDataPreprocessor.load_and_merge(*paths)

    log.info(
        "Combined dataset: %d samples × %d features",
        X.shape[0], X.shape[1],
    )
    log.info("Feature list (%d): %s", len(prep.feature_names), prep.feature_names)

    # ── Persist preprocessor state for the backend ─────────────────────
    # preprocessor_state.json  – full state (feature names + scaler)
    prep_state_path = OUTPUT_DIR / "preprocessor_state.json"
    prep.save_state(prep_state_path)
    log.info("preprocessor_state.json saved → %s", prep_state_path)

    # scaler.json – kept for backwards compatibility with the Node.js backend
    scaler_dict = {
        "mean":          prep.scaler.mean_.tolist(),
        "scale":         prep.scaler.scale_.tolist(),
        "feature_names": prep.feature_names,
    }
    scaler_path = OUTPUT_DIR / "scaler.json"
    scaler_path.write_text(json.dumps(scaler_dict, indent=2))
    log.info("scaler.json saved → %s", scaler_path)

    return X, y.to_numpy()


# ---------------------------------------------------------------------------
# DEEP LEARNING – training / evaluation helpers
# ---------------------------------------------------------------------------
def train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> nn.Module:
    model.to(DEVICE)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            log.info("  epoch %3d/%d  loss=%.4f", epoch + 1, EPOCHS, total_loss / len(loader))

    return model


def evaluate_dl_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probs = model(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy().ravel()

    y_pred = (probs > 0.5).astype(int)
    return {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":              float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
def train_models() -> None:
    X, y = load_full_dataset()
    num_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    metrics_out: Dict[str, Any] = {}

    # ── Classical ML ────────────────────────────────────────────────────
    ml_models = {
        "knn":            KNeighborsClassifier(n_neighbors=5),
        "svm":            SVC(kernel="rbf", probability=True, random_state=42),
        "logistic":       LogisticRegression(max_iter=3000, random_state=42),
        "gradient_boost": GradientBoostingClassifier(random_state=42),
    }

    for name, clf in ml_models.items():
        log.info("Training ML model: %s", name)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        metrics_out[name] = {
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
            "auc":              float(roc_auc_score(y_test, y_prob)),
            "cv_mean":          float(cv_scores.mean()),
            "cv_std":           float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        log.info("  accuracy=%.4f  AUC=%.4f", metrics_out[name]["accuracy"], metrics_out[name]["auc"])

        # Export as ONNX – input shape uses the *actual* feature count
        onnx_model = convert_sklearn(
            clf,
            initial_types=[("input", FloatTensorType([None, num_features]))],
            options={id(clf): {"zipmap": False}},
        )
        onnx_path = OUTPUT_DIR / f"{name}.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        log.info("  saved → %s", onnx_path)

    # ── Deep Learning ───────────────────────────────────────────────────
    log.info("Training deep learning models (num_features=%d)", num_features)

    # DL models expect shape (batch, 1, num_features)
    X_train_dl = X_train.reshape(-1, 1, num_features)
    X_test_dl  = X_test.reshape(-1, 1, num_features)

    dl_configs = {
        "cnn":      CNN1D(num_features),
        "cnn_lstm": CNNLSTM(num_features),
    }

    for name, model in dl_configs.items():
        log.info("Training DL model: %s", name)
        model = train_dl_model(model, X_train_dl, y_train)
        metrics_out[name] = evaluate_dl_model(model, X_test_dl, y_test)
        log.info(
            "  accuracy=%.4f  AUC=%.4f",
            metrics_out[name]["accuracy"], metrics_out[name]["auc"],
        )

        # Export as ONNX with dynamic batch size
        dummy_input = torch.randn(1, 1, num_features)
        onnx_path   = OUTPUT_DIR / f"{name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )
        log.info("  saved → %s", onnx_path)

    # ── Save all metrics ─────────────────────────────────────────────────
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    log.info("model_metrics.json saved → %s", metrics_path)

    log.info("ALL MODELS TRAINED AND EXPORTED SUCCESSFULLY")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_models()