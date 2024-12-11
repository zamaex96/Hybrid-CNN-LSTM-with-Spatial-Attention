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



class HybridCNNLSTM(nn.Module):
    def __init__(self, input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size, cnn_output_size=10):
        """
        Hybrid CNN-LSTM model for sequence classification.
        :param input_channels: Number of input channels for CNN.
        :param cnn_channels: Number of output channels for CNN layers.
        :param lstm_hidden_size: Hidden size of LSTM.
        :param lstm_num_layers: Number of LSTM layers.
        :param output_size: Number of output classes.
        :param cnn_output_size: Output size of CNN after adaptive pooling (default: 10).
        """
        super(HybridCNNLSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample sequence length
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=cnn_output_size)  # Ensure consistent output size
        )
        
        # LSTM for sequential feature modeling
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of the model for better training performance.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass for HybridCNNLSTM.
        :param x: Input tensor of shape [batch_size, sequence_length, input_channels] or [batch_size, input_channels, sequence_length].
        :return: Output tensor of shape [batch_size, output_size].
        """
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if x.dim() == 2:
            # Reshape 2D input [batch_size, seq_len] to [batch_size, 1, seq_len]
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")

        # CNN processing: [batch_size, channels, seq_len] -> [batch_size, cnn_channels*2, cnn_output_size]
        x = self.cnn(x)

        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM processing: [batch_size, seq_len, lstm_hidden_size]
        x, _ = self.lstm(x)

        # Fully connected layer: Use the last LSTM output for classification
        x = self.fc(x[:, -1, :])

        return x

