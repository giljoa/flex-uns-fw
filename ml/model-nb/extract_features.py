import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

# Parameters
window_size = int(5120/decimation_factor)  # Segment size
stride = window_size  # Non-overlapping

def extract_features(data_dict, class_labels):
    """
    Extract FFT-based features from vibration data using vectorized operations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with class labels as keys and numpy arrays (num_samples, num_sensors) as values.
    class_labels : list
        List of class labels corresponding to data_dict keys.
    
    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features, 1).
    Y : numpy.ndarray
        One-hot encoded labels of shape (n_samples, n_classes).
    
    Notes
    -----
    - Applies Hilbert transform to vibration sensors (channels 0, 3)
    - Uses Z-score normalization for all segments
    - Extracts positive frequencies only from FFT
    """
    start_time = time.time()
    print("Starting vectorized feature extraction...")
    
    all_features = []
    all_labels = []
    
    # Iterate through each class label
    for label_idx, label in enumerate(class_labels):
        print(f"Processing {label} ({label_idx + 1}/{len(class_labels)})...")
        
        if label not in data_dict:
            print(f"Warning: {label} not found in data_dict, skipping...")
            continue
            
        data = data_dict[label]  # Shape: (1,536,000, 4)
        num_samples, num_sensors = data.shape
        
        # Process each sensor channel separately
        for sensor_idx in range(num_sensors):
            sensor_data = data[:, sensor_idx]  # Extract single sensor
            
            # Vectorized segmentation
            num_segments = num_samples // window_size
            if num_segments == 0:
                print(f"Warning: Not enough samples for {label}, sensor {sensor_idx}")
                continue
                
            # Reshape for vectorized processing: (num_segments, window_size)
            segments = sensor_data[:num_segments * window_size].reshape(num_segments, window_size)
            
            # Vectorized normalization using broadcasting
            # Compute mean and std for each segment
            segment_means = np.mean(segments, axis=1, keepdims=True)
            segment_stds = np.std(segments, axis=1, keepdims=True)
            segment_stds = np.where(segment_stds == 0, 1e-8, segment_stds)  # Avoid division by zero
            
            # Z-score normalization (vectorized)
            segments_normalized = (segments - segment_means) / segment_stds
            
            # Vectorized Hilbert Transform for vibration signals (channels 0 and 3)
            if sensor_idx in [0, 3]:  # Vibration sensors
                # Apply Hilbert transform to each segment
                segments_processed = np.abs(hilbert(segments_normalized, axis=1))
            else:
                segments_processed = segments_normalized
            
            # Vectorized FFT transformation
            # Compute FFT for all segments at once
            fft_segments = np.fft.fft(segments_processed, axis=1)
            fft_features = np.abs(fft_segments)[:, :window_size//2]  # Take positive frequencies only
            
            # Store features and labels
            all_features.append(fft_features)
            all_labels.extend([label_idx] * num_segments)
    
    # Concatenate all features
    if not all_features:
        raise ValueError("No features extracted. Check if data_dict contains valid data.")
    
    X = np.vstack(all_features)  # Shape: (total_segments, window_size//2)
    Y = np.array(all_labels)
    
    # Reshape for 1D CNN: (samples, time steps, channels)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add channel dimension
    Y = to_categorical(Y, num_classes=len(class_labels))  # Convert to one-hot
    
    elapsed_time = time.time() - start_time
    print(f"Feature extraction completed in {elapsed_time:.2f} seconds")
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features each")
    
    return X, Y

def train_test_val_split(X, Y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features, n_channels).
    Y : numpy.ndarray
        Labels of shape (n_samples, n_classes) or (n_samples,).
    test_size : float, default=0.2
        Proportion for test split.
    val_size : float, default=0.1
        Proportion for validation split.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    tuple
        (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    # First split into train+val and test
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    # Adjust val_size to be a proportion of the remaining data (train+val)
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=val_adjusted, random_state=random_state)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Extract features using the vectorized version
print("="*50)
print("FEATURE EXTRACTION WITH VECTORIZED OPERATIONS")
print("="*50)

# Use the vectorized version for better performance
X, Y = extract_features(vibration_data, class_labels)

print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Split the data into train, validation, and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test = train_test_val_split(X, Y, test_size=0.2, val_size=0.1, random_state=42)

# Check shapes
assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0]
assert Y_train.shape[0] + Y_val.shape[0] + Y_test.shape[0] == Y.shape[0]
assert X_test.shape[0] == Y_test.shape[0]
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

