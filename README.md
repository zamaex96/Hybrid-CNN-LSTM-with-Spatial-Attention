# Hybrid CNN-LSTM Model with Attention for Time Series Classification

## Introduction

This documents the training and evaluation of a **Hybrid CNN-LSTM Attention model** for time series classification in a DUI detection dataset. The model combines convolutional neural networks (CNNs) for feature extraction, long short-term memory (LSTM) networks for sequential modeling, and attention mechanisms to focus on important parts of the sequence. The goal is to classify sequences into 17 different classes based on the provided normalized time-series data.

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
