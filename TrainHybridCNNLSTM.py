import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from model import HybridCNNLSTM
from data_loader import CustomDataset

model_name = "HybridCNNLSTM"
ext="TS"

# Model parameters
input_channels = 10001
cnn_channels = 24
lstm_hidden_size = 12
lstm_num_layers = 2  # Reduced number of LSTM layers for less complexity
output_size = 9
# Change batch size here
num_epochs=100
learning_Rate=0.01
batch_Size=64

# Define Hybrid CNN-LSTM Model with Attention
model = HybridCNNLSTM(input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
#def __init__(self, input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_Rate)
#optimizer = optim.Adam(model.parameters(), lr=0.001) #alternate optimizer


# Paths to training and testing data
train_csv_path = r"C:\BULabAssets\BULabProjects\BiomaterialData\dataset\Alginate\Cropped\TrainAndTest\train.csv"
test_csv_path = r"C:\BULabAssets\BULabProjects\BiomaterialData\dataset\Alginate\Cropped\TrainAndTest\test.csv"

train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)


train_data_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_Size, shuffle=False)

epochs = num_epochs
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient clipping to avoid exploding gradients for LSTM
        optimizer.step()

        epoch_train_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss /= len(train_data_loader)
    train_loss_values.append(epoch_train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    # Testing phase
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Specify output folder for saving the model, CSV file, and plots
output_folder1 = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\Models"# Replace with desired path
os.makedirs(output_folder1, exist_ok=True)  # Create folder if it doesn't exist
output_folder2 = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\CSV"  # Replace with desired path
os.makedirs(output_folder2, exist_ok=True)  # Create folder if it doesn't exist
output_folder3 = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\Plots" # Replace with desired path
os.makedirs(output_folder3, exist_ok=True)  # Create folder if it doesn't exist
# Save the model state
# Save the final model

model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'input_channels': input_channels,
        'cnn_channels': cnn_channels,
        'num_epochs': num_epochs,
        'output_size': output_size,
        'lstm_hidden_size': lstm_hidden_size,
        'learning_rate': learning_Rate,
        'lstm_num_layers': lstm_num_layers,
         'batch_size': batch_Size,
    }
}, model_path)
print(f"Model saved at {model_path}")

train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")

# Plot the loss and accuracy on the same figure
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Save plots as PNG and PDF
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
print(f"Plots saved at {png_file_path} and {pdf_file_path}")
plt.tight_layout()
plt.show()
