import pandas as pd
import numpy as np
import os
# implementation of a rolling window for the time-series dataset while keeping the sequential dependence intact
# Load your dataset
file_path = r"C:\file.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Specify the split ratio
train_ratio = 0.8  # 80% for training, 20% for testing

# Determine the split index
split_index = int(len(data) * train_ratio)

# Split the dataset
train_data = data.iloc[:split_index]  # First 80% as training data
test_data = data.iloc[split_index:]  # Last 20% as testing data

# Rolling window configuration
window_size = 10  # Number of time steps per sequence
stride = 1        # Step size for sliding the window

# Function to create rolling windows and save as DataFrame
def create_rolling_windows_as_df(data, window_size, stride):
    """
    Create rolling windows for a dataset and format as a DataFrame.

    Parameters:
    - data: pd.DataFrame, the dataset
    - window_size: int, number of time steps in each sequence
    - stride: int, step size for the rolling window

    Returns:
    - rolling_df: pd.DataFrame, formatted rolling window data with features and target
    """
    features = data.iloc[:, :-1].values  # Assuming all columns except the last are features
    labels = data.iloc[:, -1].values     # Assuming the last column is the label
    rolling_data = []
    for i in range(0, len(features) - window_size + 1, stride):
        window = features[i:i + window_size].flatten().tolist()  # Flatten the window into a single row
        target = labels[i + window_size - 1]  # Last value in the window is the target
        rolling_data.append(window + [target])  # Combine features and target
    # Create column names for features and target
    feature_columns = [f"feature_{j}" for j in range(window_size * features.shape[1])]
    rolling_df = pd.DataFrame(rolling_data, columns=feature_columns + ["target"])
    return rolling_df

# Create rolling windows for train and test datasets
train_df = create_rolling_windows_as_df(train_data, window_size, stride)
test_df = create_rolling_windows_as_df(test_data, window_size, stride)

# Specify the folder where you want to save the data
output_folder = r"C:\MLFiles"  # Replace with your desired folder path

# Ensure the folder exists, if not, create it
os.makedirs(output_folder, exist_ok=True)

# Save the train and test DataFrames as CSV files
train_csv_path = os.path.join(output_folder, 'train.csv')
test_csv_path = os.path.join(output_folder, 'test.csv')

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train dataset saved as {train_csv_path}")
print(f"Test dataset saved as {test_csv_path}")

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
