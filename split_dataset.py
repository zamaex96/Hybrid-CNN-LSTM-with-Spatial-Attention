import pandas as pd
import numpy as np
import os
# implementation of a rolling window for the time-series dataset while keeping the sequential dependence intact
# Load your dataset
file_path = r"C:\data.csv"
data = pd.read_csv(file_path)

# Specify the split ratio and configurations
train_ratio = 0.8
window_size = 10
stride = 1
chunk_size = 1000  # Adjust this based on your memory constraints

# Specify output paths
output_folder = r"C:\MLFiles"
train_csv_path = os.path.join(output_folder, 'train.csv')
test_csv_path = os.path.join(output_folder, 'test.csv')


def process_and_save_incrementally(data, window_size, stride, output_path, chunk_size=1000):
    """
    Process data in chunks and save incrementally to avoid memory issues
    """
    # Calculate number of features
    num_features = data.shape[1] - 1  # All columns except the last (target)

    # Create header
    feature_columns = [f"feature_{j}" for j in range(window_size * num_features)]
    header = feature_columns + ["target"]

    # Initialize file with header
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Process chunks and append to file
    total_rows = len(data) - window_size + 1
    chunks_processed = 0

    for chunk_start in range(0, len(data), chunk_size):
        # Calculate chunk end considering window size
        chunk_end = min(chunk_start + chunk_size + window_size, len(data))
        chunk = data.iloc[chunk_start:chunk_end]

        features = chunk.iloc[:, :-1].values
        labels = chunk.iloc[:, -1].values

        # Calculate valid windows for this chunk
        max_window_start = min(chunk_size, len(chunk) - window_size)

        # Append processed windows to file
        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(0, max_window_start, stride):
                window = features[i:i + window_size]
                target = labels[i + window_size - 1]
                row = np.concatenate([window.flatten(), [target]])
                writer.writerow(row)

        chunks_processed += max_window_start
        print(
            f"Progress: {chunks_processed}/{total_rows} rows processed ({(chunks_processed / total_rows * 100):.1f}%)")

        # Clean up to free memory
        del chunk, features, labels


# Split data
split_index = int(len(data) * train_ratio)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print("Processing training data...")
process_and_save_incrementally(
    train_data,
    window_size,
    stride,
    train_csv_path,
    chunk_size=chunk_size
)

print("\nProcessing test data...")
process_and_save_incrementally(
    test_data,
    window_size,
    stride,
    test_csv_path,
    chunk_size=chunk_size
)

# Verify file sizes and row counts
train_size = os.path.getsize(train_csv_path) / (1024 * 1024)  # Size in MB
test_size = os.path.getsize(test_csv_path) / (1024 * 1024)  # Size in MB

print(f"\nFiles created successfully:")
print(f"Train dataset saved as {train_csv_path} ({train_size:.1f} MB)")
print(f"Test dataset saved as {test_csv_path} ({test_size:.1f} MB)")

# Optional: Read the first few rows to verify the output
print("\nVerifying output (first 10 rows of training data):")
print(pd.read_csv(train_csv_path, nrows=10))
