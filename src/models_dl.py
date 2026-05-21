import torch
import torch.nn as nn

#1. simple 1D CNN Model
class CNNModel(nn.Module):
    """ A lightweight 1D CNN for heart rate estimation.
    Input shape: (batch, 512, 4)"""

    def __init__(self, num_filters=64, dropout=0.2, kernel_size=5):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2),    #512 -> 256

            nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2),    # 256 -> 128

            nn.Dropout(dropout)
        )

        #flattened size = (num_filters*2)*128
        self.fc = nn.Sequential(
            nn.Linear(num_filters*2*128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, 512, 4) -> (batch, 4, 512)
        x = x.permute(0, 2,1)
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


#2. CNN + LSTM Hybrid Model
class CNNLSTMModel(nn.Module):
    """ CNN for feature extraction + LSTM for temporal modeling.
    Input shape: (batch, 512, 4)"""

    def __init__(self, num_filters=64, lstm_hidden=64, dropout=0.2, kernel_size=5):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2),    #512 -> 256

            nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2),  #256 -> 128
        )

        #LSTM input size = num_filters*2
        self.lstm = nn.LSTM(
            input_size = num_filters*2,
            hidden_size = lstm_hidden,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self.dropout = nn.Dropout(dropout)

        #final regression layer
        self.fc = nn.Linear(lstm_hidden * 2, 1)


    def forward(self, x):
        #x: (batch, 512, 4) -> (batch, 4, 512)
        x = x.permute(0, 2, 1)

        #CNN output: (batch, C, 128)
        x = self.cnn(x)

        #LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)     #(batch, 128, C)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        #Use last timestep
        last_step = lstm_out[:, -1, :]

        return self.fc(last_step)
































        
        



























        





























        














        
