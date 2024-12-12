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
    def __init__(self, input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(HybridCNNLSTM, self).__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample sequence length
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size)  # Ensure consistent output size
        )
        # LSTM for sequential feature modeling
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        input_channels= n input_channels # insert the n number of input channels or input size or number of features
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if x.dim() == 2:
            # Reshape 2D input [batch_size, seq_len] to [batch_size, 1, seq_len]
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")

        # Ensure correct input shape for CNN [batch_size, channels, seq_len]
        if x.size(1) == 1:
            x = x.repeat(1, input_channels, 1)  # Repeat single channel if needed

        # CNN processing
        x = self.cnn(x)

        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM processing
        x, _ = self.lstm(x)

        # Use the last LSTM output for classification
        x = self.fc(x[:, -1, :])

        return x
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if x.dim() == 2:
            # Reshape 2D input [batch_size, seq_len] to [batch_size, 1, seq_len]
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")

        # Ensure correct input shape for CNN [batch_size, channels, seq_len]
        if x.size(1) == 1:
            x = x.repeat(1, input_channels, 1)  # Repeat single channel if needed

        # CNN processing
        x = self.cnn(x)

        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM processing
        x, _ = self.lstm(x)

        # Use the last LSTM output for classification
        x = self.fc(x[:, -1, :])

        return x

# Alternative of HybridCNNLSTM
class HybridCNNLSTM(nn.Module):
    def __init__(self, input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample sequence length
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Ensure consistent output size of 1 for each sequence
        )
        
        # LSTM for sequential feature modeling
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if input is 2D
        elif x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")

        # If input has only one channel, duplicate it to match `input_channels`
        if x.size(1) == 1:
            x = x.repeat(1, self.cnn[0].in_channels, 1)  # Use the first conv layer's in_channels

        # CNN processing
        x = self.cnn(x)

        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.squeeze(2)  # Remove the dimension where we have only 1 length due to AdaptiveMaxPool1d

        # LSTM processing
        x, _ = self.lstm(x.unsqueeze(1))  # Add sequence length dimension for LSTM

        # Use the last LSTM output for classification
        x = self.fc(x.squeeze(1))  # Remove the sequence length dimension

        return x
