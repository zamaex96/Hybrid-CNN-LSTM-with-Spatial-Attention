# Hybrid CNN-LSTM Model with Attention for Time Series Classification

## Introduction

This documents the training and evaluation of a **Hybrid CNN-LSTM Attention model** for time series classification in a dataset. The model combines convolutional neural networks (CNNs) for feature extraction, long short-term memory (LSTM) networks for sequential modeling, and attention mechanisms to focus on important parts of the sequence. The goal is to classify sequences into different classes based on the provided normalized time-series data.

## Model Architecture

The hybrid model consists of:

1. **CNN Layers**: Extract spatial features from the time series.
   - Two convolutional layers (`Conv1d`) with ReLU activations.
   - Two max-pooling layers (`MaxPool1d`) for downsampling.
   
2. **Attention Mechanism**: A spatial attention mechanism highlights the important parts of the sequence, enhancing the model's ability to focus on critical segments of the input.

3. **LSTM Layers**: Process the sequential data to capture temporal dependencies.
   - LSTM with 12 layers and a hidden size of 8.

4. **Fully Connected (FC) Layer**: Used to map the output of the LSTM to 17 class labels.

The **model parameters** are:
- Input size: 12
- CNN Channels: 16
- LSTM Hidden Size: 8
- LSTM Layers: 12
- Output Size: 17

## Dataset

The dataset consists of time series data that has been normalized. Separate CSV files are used for training and testing:
- **Training Data**: `train.csv`
- **Testing Data**: `test.csv`

The dataset is loaded using a custom `CustomDataset` class and fed into the model using `DataLoader`. Each batch size is set to 1.

## Training Process

The model is trained for 200 epochs with the following configurations:
- **Optimizer**: SGD (Stochastic Gradient Descent) with a learning rate of 0.01.
- **Loss Function**: Cross-Entropy Loss to handle multi-class classification.

### Key Steps in the Training Loop:
1. **Training Phase**: 
   - The model's weights are updated based on training data using backpropagation.
   - Training loss and accuracy are calculated after each epoch.

2. **Testing Phase**: 
   - The model is evaluated on the test dataset.
   - Testing loss and accuracy are tracked.

The training and testing loss/accuracy values are stored after each epoch for later analysis.

## Results

After training for 200 epochs, the model's performance is as follows:

- **Final Training Accuracy**: Varies based on specific epoch values.
- **Final Testing Accuracy**: Varies based on specific epoch values.

The loss and accuracy metrics are plotted to visualize the training progress.

### Performance Metrics

The performance metrics captured during training are:
- **Training Loss**: Shows how well the model fits the training data.
- **Testing Loss**: Indicates how well the model generalizes to unseen data.
- **Training Accuracy**: Measures the percentage of correctly classified training examples.
- **Testing Accuracy**: Measures the percentage of correctly classified testing examples.

## Visualizations

Plots of the training and testing loss/accuracy over the epochs are generated.

### Loss Plot:
- **Training Loss**: The trend generally decreases as the model learns from the data.
- **Testing Loss**: A steady decline (or plateau) is expected if the model generalizes well.

### Accuracy Plot:
- **Training Accuracy**: Typically increases over time as the model improves.
- **Testing Accuracy**: Should follow a similar pattern to training accuracy if the model generalizes well.

Both plots are saved in PNG and PDF formats for documentation purposes.

## Model and Data Storage

- **Model**: Saved in `.pth` format for future inference or fine-tuning. 
- **Training Data**: Saved as a CSV file containing the loss and accuracy values across all epochs.

The files are stored in predefined output folders:
- **Model Path**: `Models/HybridCNNLSTMAttention_TSNormOnly.pth`
- **Training Data CSV Path**: `CSV/HybridCNNLSTMAttention_TSNormOnly.csv`
- **Plot Files**: 
  - PNG: `Plots/HybridCNNLSTMAttention_TS.png`
  - PDF: `Plots/HybridCNNLSTMAttention_TS.pdf`

## Conclusion

The **Hybrid CNN-LSTM with Attention** architecture successfully processes time series data for multi-class classification tasks. Both the training and evaluation processes are automated, and the results are well-documented through plots and saved models.

### Future Work:
- **Hyperparameter Tuning**: Explore different values for learning rates, batch sizes, and CNN/LSTM parameters.
- **Advanced Attention Mechanisms**: Experiment with more complex attention mechanisms to improve model accuracy.
- **Regularization**: Add dropout or weight decay to reduce overfitting and improve generalization.

This model forms a strong foundation for time series classification in DUI detection, and further optimizations may enhance performance.

--- 

# Step-by-Step Process of the Code

The following steps describe the process and functionality of the provided code, which builds, trains, and evaluates a **Hybrid CNN-LSTM Attention model** for time series classification.

## 1. **Importing Required Libraries**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from model import HybridCNNLSTMAttention
from data_loader import CustomDataset
```

The code begins by importing the necessary libraries:
- `torch`, `torch.nn`, `torch.optim`: For deep learning operations and model training.
- `matplotlib` and `pandas`: For plotting training/testing metrics and handling data.
- `os`: For file and folder operations.
- `HybridCNNLSTMAttention` and `CustomDataset` are imported from custom files `model.py` and `data_loader.py`.

## 2. **Defining Model and File Paths**

```python
model_name = "HybridCNNLSTMAttention"
ext = "TSNormOnly"
```

- `model_name` specifies the name of the model.
- `ext` is used as an additional identifier for file naming.

### Model Parameters:
```python
input_size = 1201
cnn_channels = 16
lstm_hidden_size = 8
lstm_num_layers = 12
output_size = 17
```
- **input_size**: The number of features in each time series input.
- **cnn_channels**: The number of channels in the first CNN layer.
- **lstm_hidden_size**: Hidden size of the LSTM layers.
- **lstm_num_layers**: Number of stacked LSTM layers.
- **output_size**: Number of output classes (17 in this case).

### Initialize Model:

```python
model = HybridCNNLSTMAttention(input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
```

- The `HybridCNNLSTMAttention` model is initialized using the defined parameters.

### Set Device (CPU or GPU):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

- The model is moved to GPU if available, otherwise it uses the CPU.

## 3. **Defining Loss Function and Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

- **Loss Function**: Cross-entropy loss is used as the classification criterion.
- **Optimizer**: Stochastic Gradient Descent (SGD) is used with a learning rate of `0.01` for updating the model parameters during training.

## 4. **Loading the Dataset**

### Paths to Training and Testing Data:

```python
train_csv_path = r"train_norm_only.csv"
test_csv_path = r"test_norm_only.csv"
```

- **train_csv_path**: Path to the training dataset (normalized time series).
- **test_csv_path**: Path to the testing dataset.

### Create Dataset Objects:

```python
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)
```

- **train_dataset** and **test_dataset** are instances of the `CustomDataset` class for loading the training and testing data.

### Create DataLoader Objects:

```python
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

- **train_data_loader**: Loads the training data in batches of size `1` and shuffles the data.
- **test_data_loader**: Loads the testing data in batches of size `1` without shuffling.

## 5. **Training and Testing Loop**

### Initialize Lists for Storing Metrics:
```python
epochs = 200
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []
```

- These lists will store loss and accuracy values for each epoch during training and testing.

### Epoch Loop:

```python
for epoch in range(epochs):
```

- A loop that runs for 200 epochs to train the model. Each epoch consists of two phases: training and testing.

### Training Phase:
```python
model.train()
epoch_train_loss = 0.0
correct_train = 0
total_train = 0
```

- The model is set to training mode using `model.train()`.
- Initialize the loss and accuracy counters for the current epoch.

#### Mini-Batch Training:
```python
for inputs, labels in train_data_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

- The inputs and labels are moved to the selected device (CPU/GPU).
- The model processes the input and calculates the loss.
- The optimizer updates the model weights using backpropagation.

#### Track Accuracy and Loss:
```python
epoch_train_loss += loss.item()
_, predicted_train = torch.max(outputs.data, 1)
total_train += labels.size(0)
correct_train += (predicted_train == labels).sum().item()
```

- The total training loss and the number of correct predictions are tracked for each batch.
- After completing all batches in the epoch, the loss and accuracy are averaged and stored.

### Testing Phase:

```python
model.eval()
epoch_test_loss = 0.0
correct_test = 0
total_test = 0
```

- The model is set to evaluation mode using `model.eval()`.
- The loss and accuracy for testing data are tracked similarly to the training phase, but without backpropagation (`torch.no_grad()`).

### Store Loss and Accuracy:
```python
train_loss_values.append(epoch_train_loss)
train_accuracy_values.append(train_accuracy)
test_loss_values.append(epoch_test_loss)
test_accuracy_values.append(test_accuracy)
```

- Loss and accuracy values for both training and testing phases are stored in the respective lists.

### Print Epoch Results:

```python
if (epoch + 1) % 5 == 0:
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
```

- Results are printed every 5 epochs, displaying the training and testing loss and accuracy.

## 6. **Saving the Model and Data**

### Create Output Folders:
```python
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)
```

- These lines ensure that the specified output directories exist, and create them if they don't.

### Save Model:
```python
model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save(model.state_dict(), model_path)
```

- The trained model's weights are saved in a `.pth` file for future inference or further training.

### Save Training Data:
```python
train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
```

- The collected training and testing loss/accuracy values are saved in a CSV file for further analysis.

## 7. **Plotting the Results**

### Create Plot for Loss and Accuracy:
```python
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
```

- A figure is created with two subplots: one for loss and one for accuracy over epochs.

### Save the Plot:
```python
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
```

- The plots are saved as both PNG and PDF files.

## 8. **Displaying the Plot**

```python
plt.tight_layout()
plt.show()
```

- The layout is adjusted to avoid overlap, and the plot is displayed.

---

This step-by-step breakdown explains the entire process of building, training, testing, saving, and visualizing the performance of the **Hybrid CNN-LSTM Attention model**.
