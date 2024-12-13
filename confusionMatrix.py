import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np  # Add this line to import numpy
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import *
from data_loader import *
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
#from sklearn.utils.class_weight import compute_class_weight
#import random

model_name = "HybridCNNLSTM"
ext="CM-comb"
class_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
delta_font_size=-10
fixed_size=5
output_folder1 = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\Plots"
os.makedirs(output_folder1, exist_ok=True)
model_path = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\Models\HybridCNNLSTM_TS-comb2.pth"
test_csv_path = r"C:\BULabAssets\BULabProjects\BiomaterialData\dataset\Alginate\Cropped\CombinedTrainAndTest\test.csv"

# Load the checkpoint
checkpoint = torch.load(model_path,weights_only=True)
# Default model parameters
default_params = {
    'input_channels': 10001,
    'hidden_size': 112,
    'output_size': 9,
    'num_layers': 8,
    'n_head': 2,
    'cnn_channels': 24,
    'lstm_hidden_size': 12,
    'lstm_num_layers': 2,
    'batch_size': 64,
    'epochs': 200
}
# Check if hyperparameters were saved along with the model
if 'hyperparameters' in checkpoint:
    hyperparams = checkpoint['hyperparameters']
    print("Saved Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
else:
    print("No hyperparameters were found in the checkpoint.")

# Unpack the updated parameters
input_channels = default_params['input_channels']
hidden_size = default_params['hidden_size']
output_size = default_params['output_size']
num_layers = default_params['num_layers']
n_head = default_params['n_head']
cnn_channels = default_params['cnn_channels']
lstm_hidden_size = default_params['lstm_hidden_size']
lstm_num_layers = default_params['lstm_num_layers']
batch_size = default_params['batch_size']
#epochs = default_params['num_epochs']
epochs = default_params['epochs']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device,weights_only=True)
print(checkpoint.keys())  # Check the keys to see if it matches expected  keys


# Function to compute the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    accuracy = accuracy_score(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=0.8, linecolor='black',
                     cbar_kws={'label': 'Percentage'},
                     annot_kws={"size": 18 + delta_font_size, "weight": "bold"})
    # Increase colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Percentage', fontsize=18 + delta_font_size + fixed_size)
    # Increase colorbar tick label size
    cbar.ax.tick_params(labelsize=16 + delta_font_size + fixed_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Percentage', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')

    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', fontsize=20 + delta_font_size + fixed_size,
              fontweight='bold')
    plt.xlabel('Predicted', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    plt.ylabel('True', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    plt.xticks(fontsize=16 + delta_font_size + fixed_size, rotation=0, fontweight='bold')
    plt.yticks(fontsize=16 + delta_font_size + fixed_size, rotation=0, fontweight='bold')
    png_file_path = os.path.join(output_folder1, f"{model_name}_{ext}.png")
    plt.savefig(png_file_path, format='png', dpi=1500)
    plt.show()


model_inference = HybridCNNLSTM(input_channels, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
#model_inference = load_model_from_checkpoint(model_path)
#model_inference = SpatialAttentionModel(input_size, hidden_size, output_size, num_layers)
#model_inference = TransformerModel(input_size, hidden_size, output_size, num_layers=num_layers, nhead=n_head)
#model_inference = HybridCNNLSTMAttention(input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'),weights_only=True)
model_inference.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inference.to(device)
print(f"Using device: {device}")


test_dataset = CustomDataset(test_csv_path)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_inference.eval()
true_labels = []
predicted_labels = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model_inference(inputs)
        print(f"Model output shape: {outputs.shape}")  # Print the shape of outputs: (batch_size, numberOfOutputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())



# Calculate and print accuracy
accuracy = 100 * correct / total
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Print per-class accuracy
print("\nPer-class accuracy:")
cm = confusion_matrix(true_labels, predicted_labels)
for i, class_name in enumerate(class_names):
    class_correct = cm[i, i]
    class_total = cm[i].sum()
    class_acc = class_correct / class_total * 100
    print(f"{class_name}: {class_acc:.2f}%")

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100

# Print metrics
print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# Save metrics to text file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder2 = r"C:\BULabAssets\BULabProjects\BiomaterialData\ML\Metrics"
os.makedirs(output_folder2, exist_ok=True)
metrics_filename = os.path.join(output_folder2, f"model_metrics_{model_name}_{ext}_{timestamp}.txt")

with open(metrics_filename, 'w') as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Detail: {ext}\n")
    f.write(f"Number of Classes: {output_size}\n\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write(f"Precision (macro): {precision:.2f}%\n")
    f.write(f"Recall (macro): {recall:.2f}%\n")
    f.write(f"F1 Score (macro): {f1:.2f}%\n\n")

    # Detailed per-class accuracy
    f.write("Per-class Accuracy:\n")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_acc = class_correct / class_total * 100
        f.write(f"{class_name}: {class_acc:.2f}%\n")

print(f"\nMetrics saved to {metrics_filename}")

plot_confusion_matrix(true_labels, predicted_labels, class_names)
