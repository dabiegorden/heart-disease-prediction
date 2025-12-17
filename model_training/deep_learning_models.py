import torch
import torch.nn as nn

# ============================================================
# 1D CNN MODEL
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, features)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# CNN + LSTM MODEL
# ============================================================
class CNNLSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, features)
        x = self.conv(x)               # (batch, 32, features)
        x = x.permute(0, 2, 1)         # (batch, features, 32)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
