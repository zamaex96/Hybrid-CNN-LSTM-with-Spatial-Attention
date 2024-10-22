import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd

class HybridCNNLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(HybridCNNLSTMAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),  # Pooling 1, Use smaller kernel_size and/or stride, or remove one pooling layer if necessary.
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=16)  # Adaptive pooling to ensure valid output size
        )
        self.attention = SpatialAttention(cnn_channels * 2)
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            # If input is 2D, assume shape is [batch_size, seq_len], reshape to [batch_size, 1, seq_len]
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")

        # Change shape to [batch_size, input_size, seq_len] for CNNs
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # Attention expects [batch_size, channels, seq_len]
        x = self.attention(x)

        # Change shape to [batch_size, seq_len, cnn_channels * 2] for LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last LSTM output for classification
        return x


class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=1)

    def forward(self, x):
        attn_weights = torch.softmax(self.conv(x), dim=-1)
        return x * attn_weights
