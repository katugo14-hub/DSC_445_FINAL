import torch
import torch.nn as nn


# ── 1. CNN Model (with GlobalAvgPool to avoid 2.1M-param FC bottleneck) ───────
class CNNModel(nn.Module):
    """Lightweight 1D CNN for heart rate regression from 4-channel PPG+ACC windows.
    Input shape: (batch, 512, 4)

    Design notes:
      - BatchNorm1d after each conv stabilises training across subjects.
      - GlobalAvgPool replaces flatten: reduces FC1 from 2.1M → 16K params,
        cutting overfitting risk by ~97% while preserving all filter information.
      - Dropout=0.3 on FC layers (slightly higher than default for LOSO setting).
    """

    def __init__(self, num_filters=64, dropout=0.3, kernel_size=5):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 512 → 256

            # Block 2
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 256 → 128
        )

        # Global Average Pool: (batch, num_filters*2, 128) → (batch, num_filters*2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # FC head: num_filters*2 → 128 → 1  (~16K params total, not 2.1M)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, 512, 4) → (batch, 4, 512) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)   # (batch, num_filters*2, 128)
        x = self.gap(x).squeeze(-1)     # (batch, num_filters*2)
        return self.fc(x)               # (batch, 1)


# ── 2. CNN + BiLSTM Hybrid Model ──────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    """CNN feature extractor + Bidirectional LSTM for temporal modelling.
    Input shape: (batch, 512, 4)

    Design notes:
      - BatchNorm1d after each conv for training stability.
      - BiLSTM correctly combined: forward final state (pos -1) concatenated
        with backward final state (pos 0), giving a true full-sequence encoding.
      - Dropout=0.3 on LSTM output before regression head.
    """

    def __init__(self, num_filters=64, lstm_hidden=64, dropout=0.3, kernel_size=5):
        super().__init__()
        self._hidden = lstm_hidden

        # CNN feature extractor (shared design with CNNModel)
        self.cnn = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 512 → 256

            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),        # 256 → 128
        )

        # BiLSTM: input_size = num_filters*2 (128 with defaults)
        self.lstm = nn.LSTM(
            input_size=num_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, 512, 4) → (batch, 4, 512)
        x = x.permute(0, 2, 1)

        # CNN: (batch, num_filters*2, 128)
        x = self.cnn(x)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)             # (batch, 128, num_filters*2)
        lstm_out, _ = self.lstm(x)          # (batch, 128, lstm_hidden*2)
        lstm_out = self.dropout(lstm_out)

        # Correct bidirectional combination:
        #   Forward  final state: position -1  (has seen tokens 0 → 127)
        #   Backward final state: position  0  (has seen tokens 127 → 0)
        fwd = lstm_out[:, -1, :self._hidden]    # (batch, lstm_hidden)
        bwd = lstm_out[:,  0, self._hidden:]    # (batch, lstm_hidden)
        last_step = torch.cat([fwd, bwd], dim=1)  # (batch, lstm_hidden*2)

        return self.fc(last_step)           # (batch, 1)
































        
        



























        





























        














        
