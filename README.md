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
   - LSTM with 1 layers and a hidden size of 8.

4. **Fully Connected (FC) Layer**: Used to map the output of the LSTM to 4 class labels.

The **model parameters** are:
- Input size: 8
- CNN Channels: 16
- LSTM Hidden Size: 8
- LSTM Layers: 1
- Output Size: 4

## Dataset

The dataset consists of time series data that has been normalized. Separate CSV files are used for training and testing:
- **Training Data**: `train.csv`
- **Testing Data**: `test.csv`

The dataset is loaded using a custom `CustomDataset` class and fed into the model using `DataLoader`. Each batch size is set to 32.

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
- **Model Path**: `Models/HybridCNNLSTMAttention_TS.pth`
- **Training Data CSV Path**: `CSV/HybridCNNLSTMAttention_TS.csv`
- **Plot Files**: 
  - PNG: `Plots/HybridCNNLSTMAttention_TS.png`
  - PDF: `Plots/HybridCNNLSTMAttention_TS.pdf`
    
### save_model Function

#### Purpose
The `save_model` function saves a PyTorch model and its associated hyperparameters to a specified file location. This allows for easy storage and later retrieval of the trained model and its configuration for inference or further training.

#### Parameters
- **output_folder (str):** Path to the directory where the model file will be saved.
- **model_name (str):** Name of the model file (without extension).
- **ext (str):** File extension for the saved file (e.g., 'checkpoint', 'final').
- **model (torch.nn.Module):** The PyTorch model instance to be saved.
- **input_size (int):** Size of the input feature vector.
- **cnn_channels (int):** Number of channels in the CNN layers of the model.
- **num_epochs (int):** Total number of epochs used for training.
- **output_size (int):** Dimension of the model's output.
- **lstm_hidden_size (int):** Number of hidden units in the LSTM layers.
- **learning_rate (float):** Learning rate used for model training.
- **lstm_num_layers (int):** Number of stacked LSTM layers.
- **batch_size (int):** Batch size used for training.

#### Returns
- **str:** The file path where the model was saved.

#### Key Operations
1. Constructs a file path using the provided folder, model name, and extension.
2. Saves the model's `state_dict` and hyperparameters into a `.pth` file using `torch.save`.
3. Prints and returns the path of the saved model.

#### Example Usage
```python
model_path = save_model(
    output_folder="models/",
    model_name="my_model",
    ext="checkpoint",
    model=my_model,
    input_size=8,
    cnn_channels=64,
    num_epochs=50,
    output_size=4,
    lstm_hidden_size=256,
    learning_rate=0.001,
    lstm_num_layers=2,
    batch_size=32
)
print(f"Model saved at: {model_path}")
```
## Conclusion

The **Hybrid CNN-LSTM with Attention** architecture successfully processes time series data for multi-class classification tasks. Both the training and evaluation processes are automated, and the results are well-documented through plots and saved models.

### Future Work:
- **Hyperparameter Tuning**: Explore different values for learning rates, batch sizes, and CNN/LSTM parameters.
- **Advanced Attention Mechanisms**: Experiment with more complex attention mechanisms to improve model accuracy.
- **Regularization**: Add dropout or weight decay to reduce overfitting and improve generalization.

This model forms a strong foundation for time series classification, and further optimizations may enhance performance.

--- 

# Step-by-Step Process 

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
ext = "TS"
```

- `model_name` specifies the name of the model.
- `ext` is used as an additional identifier for file naming.

### Model Parameters:
```python
input_size = 8
cnn_channels = 16
lstm_hidden_size = 8
lstm_num_layers = 1
output_size = 4
```
- **input_size**: The number of features in each time series input.
- **cnn_channels**: The number of channels in the first CNN layer.
- **lstm_hidden_size**: Hidden size of the LSTM layers.
- **lstm_num_layers**: Number of stacked LSTM layers.
- **output_size**: Number of output classes (4 in this case).

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

### Alternatives to Cross-Entropy Loss and Optimizers:

#### **1. Alternatives to Cross-Entropy Loss**
Cross-entropy loss is widely used for classification problems, but other loss functions may be suitable based on your task:

- **Focal Loss**
  - Suitable for imbalanced datasets.
  - Focuses more on hard-to-classify samples by down-weighting easy samples.
  - Implementation: Available in libraries like PyTorch or custom implementations.

- **Mean Squared Error (MSE)**
  - Typically used for regression but can be applied to classification when one-hot encoding is used.
  - Not ideal for classification as it treats probabilities linearly.

- **Kullback-Leibler Divergence Loss (KLDivLoss)**
  - Measures the divergence between two probability distributions.
  - Useful when comparing soft labels or probabilistic outputs.

- **Hinge Loss**
  - Commonly used for binary classification tasks with Support Vector Machines (SVMs).
  - Encourages a margin of separation between classes.

- **Label Smoothing**
  - A variation of cross-entropy loss that smooths target labels to prevent overconfidence.
  - Useful in cases prone to overfitting or noisy labels.

- **Binary Cross-Entropy (BCE)**
  - Specialized for binary classification tasks.
  - Can also be extended to multi-label classification problems.

- **Contrastive Loss**
  - Useful in tasks like face recognition or similarity learning.
  - Operates on pairs of samples to measure the similarity or dissimilarity.

---

#### **2. Alternatives to Stochastic Gradient Descent (SGD) Optimizer**
Depending on the nature of your problem and dataset, alternative optimizers may provide better convergence:

- **Adam (Adaptive Moment Estimation)**
  - Combines the advantages of RMSProp and momentum.
  - Well-suited for sparse data and non-stationary objectives.
  - Common usage: `optim.Adam(model.parameters(), lr=0.001)`

- **AdamW (Adam with Weight Decay Regularization)**
  - Variation of Adam with improved weight decay regularization.
  - Helps prevent overfitting.
  - Common usage: `optim.AdamW(model.parameters(), lr=0.001)`

- **RMSProp (Root Mean Square Propagation)**
  - Divides the learning rate by a running average of the magnitudes of recent gradients.
  - Well-suited for recurrent neural networks (RNNs).
  - Common usage: `optim.RMSprop(model.parameters(), lr=0.001)`

- **Adagrad (Adaptive Gradient Algorithm)**
  - Adapts learning rates based on historical gradient information.
  - Suitable for sparse data or parameters.
  - Common usage: `optim.Adagrad(model.parameters(), lr=0.001)`

- **Adadelta**
  - Addresses some limitations of Adagrad by restricting step size.
  - Common usage: `optim.Adadelta(model.parameters(), lr=1.0)`

- **NAdam (Nesterov-accelerated Adam)**
  - Extends Adam by incorporating Nesterov momentum.
  - Common usage: `optim.NAdam(model.parameters(), lr=0.001)`

- **LBFGS (Limited-memory BFGS)**
  - A quasi-Newton method optimizer.
  - Suitable for smaller datasets and optimization problems with second-order behavior.
  - Common usage: `optim.LBFGS(model.parameters(), lr=0.1)`

---

### Selecting Alternatives:
- **Classification Tasks**: 
  - Use Focal Loss or Label Smoothing if data is imbalanced.
  - Use Hinge Loss for binary classification with margin-based separation.

- **Optimizers for Stability**:
  - Adam and AdamW are generally more stable for deep learning tasks.
  - RMSProp is preferred for RNNs or non-stationary datasets.

- **Fine-Tuning Hyperparameters**:
  - Experiment with learning rates, momentum, and weight decay to adapt optimizers to your dataset.

### Example:
```python
# Alternative Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Or FocalLoss(), KLDivLoss(), etc.
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Or RMSProp, Adagrad
```
  
### Table of Alternatives to Cross-Entropy Loss and SGD Optimizer

| **Type**               | **Name**              | **Description**                                                                                                                                   | **Implementation (PyTorch)**                                                                                     |
|------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Loss Functions**     | **Cross-Entropy Loss** | Default for classification tasks.                                                                                                                 | `nn.CrossEntropyLoss()`                                                                                          |
|                        | **Focal Loss**         | Focuses on hard-to-classify samples; reduces the influence of easy samples.                                                                       | [Focal Loss Implementation](https://github.com/AdeelH/pytorch-multi-class-focal-loss)                            |
|                        | **Mean Squared Error** | Regression-based loss, less common for classification.                                                                                           | `nn.MSELoss()`                                                                                                   |
|                        | **KL Divergence Loss** | Measures the divergence between predicted and target distributions.                                                                               | `nn.KLDivLoss()`                                                                                                 |
|                        | **Hinge Loss**         | Encourages a margin of separation between classes; used in SVMs.                                                                                  | `nn.HingeEmbeddingLoss()`                                                                                        |
|                        | **Label Smoothing**    | Reduces overconfidence by smoothing target labels.                                                                                                | `nn.CrossEntropyLoss(label_smoothing=0.1)` (PyTorch 1.10+)                                                       |
|                        | **Binary Cross-Entropy** | For binary or multi-label classification.                                                                                                        | `nn.BCELoss()` or `nn.BCEWithLogitsLoss()`                                                                       |
|                        | **Contrastive Loss**   | Used for similarity or metric learning tasks.                                                                                                    | Custom: See [Contrastive Loss Implementation](https://omoindrot.github.io/triplet-loss)                          |
| **Optimizers**         | **SGD**               | Basic optimizer with momentum.                                                                                                                    | `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`                                                           |
|                        | **Adam**              | Combines RMSProp and momentum; adapts learning rates.                                                                                            | `optim.Adam(model.parameters(), lr=0.001)`                                                                       |
|                        | **AdamW**             | Adam with decoupled weight decay for better regularization.                                                                                       | `optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`                                                  |
|                        | **RMSProp**           | Scales learning rates based on recent gradient magnitudes; good for RNNs.                                                                        | `optim.RMSprop(model.parameters(), lr=0.001)`                                                                    |
|                        | **Adagrad**           | Adapts learning rates for parameters with infrequent updates.                                                                                    | `optim.Adagrad(model.parameters(), lr=0.01)`                                                                     |
|                        | **Adadelta**          | Improves Adagrad by limiting step sizes for better stability.                                                                                     | `optim.Adadelta(model.parameters(), lr=1.0)`                                                                     |
|                        | **NAdam**             | Combines Adam and Nesterov momentum for faster convergence.                                                                                      | `optim.NAdam(model.parameters(), lr=0.001)`                                                                      |
|                        | **LBFGS**             | Quasi-Newton method for small datasets or second-order optimization.                                                                              | `optim.LBFGS(model.parameters(), lr=0.1)`                                                                        |

### Example Code Snippet

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example Loss Function: Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Example Optimizer: AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Focal Loss Example (Custom Implementation)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss

criterion = FocalLoss(alpha=0.25, gamma=2)
```

### Notes
- Use **Cross-Entropy Loss** for most classification tasks, unless specific challenges like class imbalance or noisy labels exist.
- Use **AdamW** or **RMSProp** as alternatives to SGD for better convergence in deep learning tasks.

## 4. **Loading the Dataset**

### Paths to Training and Testing Data:

```python
train_csv_path = r"train.csv"
test_csv_path = r"test.csv"
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

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>


