"""
retrain.py
==========
Command-line script that lets any hospital retrain a specific model on
their own dataset – regardless of which columns it contains.

Usage examples
--------------
# Retrain gradient boosting on a new hospital CSV
python retrain.py --model-type gradient_boost --data-path /data/hospital_b.csv

# Retrain CNN on an Excel file, 80 epochs, save to a custom directory
python retrain.py --model-type cnn1d --data-path data.xlsx --epochs 80 --output-dir ./my_models

Supported model types
---------------------
  logistic_regression | svm | gradient_boost | knn | cnn1d | cnn_lstm

Output per run
--------------
  <output_dir>/<model_type>_model.pkl   or  .pth  (DL)
  <output_dir>/<model_type>_preprocessor.json     (feature list + scaler)
  <output_dir>/<model_type>_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys
import os

# Add model_training folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_training")))

from data_preprocessing import CardiovascularDataPreprocessor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional deep-learning imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ML_TYPES = {"logistic_regression", "svm", "gradient_boost", "knn"}
DL_TYPES = {"cnn1d", "cnn_lstm"}


# ---------------------------------------------------------------------------
# Deep-learning model definitions
# (mirror of deep_learning_models.py – kept self-contained so this script
#  can be dropped into any project without the full package)
# ---------------------------------------------------------------------------
if DL_AVAILABLE:
    class CNN1D(nn.Module):
        def __init__(self, num_features: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.conv(x).view(x.size(0), -1)
            return self.fc(x)

    class CNNLSTM(nn.Module):
        def __init__(self, num_features: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            )
            self.lstm = nn.LSTM(32, 32, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(16, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.conv(x).permute(0, 2, 1)
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------
def train_ml(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    clf_map = {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=42),
        "svm":                 SVC(kernel="rbf", probability=True, random_state=42),
        "gradient_boost":      GradientBoostingClassifier(random_state=42),
        "knn":                 KNeighborsClassifier(n_neighbors=5),
    }
    clf = clf_map[model_type]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return clf, compute_metrics(y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# DL training
# ---------------------------------------------------------------------------
def train_dl(
    model_type: str,
    num_features: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
) -> Tuple[Any, Dict[str, float]]:
    if not DL_AVAILABLE:
        raise RuntimeError("PyTorch not installed – cannot train deep learning models.")

    # Reshape to (batch, 1, features) for Conv1d
    X_tr = X_train.reshape(-1, 1, num_features)
    X_te = X_test.reshape(-1, 1, num_features)

    model_cls = {"cnn1d": CNN1D, "cnn_lstm": CNNLSTM}[model_type]
    model = model_cls(num_features)

    dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader    = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            log.info("  epoch %d/%d", epoch + 1, epochs)

    model.eval()
    with torch.no_grad():
        probs  = model(torch.tensor(X_te, dtype=torch.float32)).numpy().ravel()
    y_pred = (probs > 0.5).astype(int)
    return model, compute_metrics(y_test, y_pred, probs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain a heart-disease model on a custom dataset."
    )
    parser.add_argument(
        "--model-type", required=True,
        choices=sorted(ML_TYPES | DL_TYPES),
        help="Type of model to train.",
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Path to training data (CSV or Excel).",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory where model artefacts are saved (default: ./models).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for deep-learning models (default: 50).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data used for testing (default: 0.2).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ─────────────────────────────────────────────
    log.info("Loading dataset: %s", args.data_path)
    prep = CardiovascularDataPreprocessor()
    X, y = prep.fit_transform(args.data_path)
    y_np = y.to_numpy()

    num_features = X.shape[1]
    log.info("Dataset: %d samples × %d features", X.shape[0], num_features)
    log.info("Features: %s", prep.feature_names)
    log.info("Class distribution: %s", dict(zip(*np.unique(y_np, return_counts=True))))

    # Save preprocessor state (includes feature list + scaler)
    prep_path = output_dir / f"{args.model_type}_preprocessor.json"
    prep.save_state(prep_path)
    log.info("Preprocessor state saved → %s", prep_path)

    # ── Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=args.test_size, random_state=42, stratify=y_np
    )

    # ── Train ─────────────────────────────────────────────────────────
    log.info("Training %s ...", args.model_type)

    if args.model_type in ML_TYPES:
        model, metrics = train_ml(args.model_type, X_train, y_train, X_test, y_test)
        model_path = output_dir / f"{args.model_type}_model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

    else:  # DL
        model, metrics = train_dl(
            args.model_type, num_features,
            X_train, y_train, X_test, y_test,
            epochs=args.epochs,
        )
        model_path = output_dir / f"{args.model_type}_model.pth"
        torch.save(
            {"state_dict": model.state_dict(), "num_features": num_features},
            model_path,
        )

    log.info("Model saved → %s", model_path)

    # ── Metrics ───────────────────────────────────────────────────────
    log.info("Results: %s", json.dumps(metrics, indent=2))

    result = {
        "success":       True,
        "model_type":    args.model_type,
        "num_features":  num_features,
        "feature_names": prep.feature_names,
        "metrics":       metrics,
        "model_path":    str(model_path),
        "prep_path":     str(prep_path),
        "num_samples":   int(X.shape[0]),
    }

    metrics_path = output_dir / f"{args.model_type}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    log.info("Metrics saved → %s", metrics_path)

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()