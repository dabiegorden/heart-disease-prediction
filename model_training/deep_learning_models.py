"""
deep_learning_models.py
=======================
Flexible CNN1D and CNN-LSTM models that accept any number of input features.
The architecture adapts dynamically at construction time so hospitals can
supply datasets with different column sets without touching this file.
"""

import torch
import torch.nn as nn


# ============================================================
# 1D CNN MODEL
# ============================================================
class CNN1D(nn.Module):
    """
    1-D Convolutional Neural Network for tabular / sequential health data.

    Parameters
    ----------
    num_features : int
        Number of input features (determined at runtime from the dataset).
    """

    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # AdaptiveAvgPool collapses the feature dimension to 1 regardless
            # of how many features are in the input – this is what makes the
            # model accept *any* column count.
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, num_features)
        x = self.conv(x)           # → (batch, 64, 1)
        x = x.view(x.size(0), -1)  # → (batch, 64)
        return self.fc(x)          # → (batch, 1)


# ============================================================
# CNN + LSTM MODEL
# ============================================================
class CNNLSTM(nn.Module):
    """
    Hybrid 1-D CNN → LSTM model for health tabular data.

    The CNN extracts local feature interactions; the LSTM models sequential
    dependencies across feature positions.  AdaptiveAvgPool is *not* used
    here – instead, each feature position becomes a time-step for the LSTM,
    so the model still works with any feature count.

    Parameters
    ----------
    num_features : int
        Number of input features (determined at runtime from the dataset).
    """

    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        # Each feature position is a time-step with 32 channels from the CNN.
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, num_features)
        x = self.conv(x)              # → (batch, 32, num_features)
        x = x.permute(0, 2, 1)        # → (batch, num_features, 32)  [time, features]
        _, (hn, _) = self.lstm(x)     # hn: (1, batch, 32)
        return self.fc(hn[-1])        # → (batch, 1)