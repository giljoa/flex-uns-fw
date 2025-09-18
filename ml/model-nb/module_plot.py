import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import hilbert
from scipy.fftpack import fft

def plot_signal(data_dict, fs, seconds=0.1, channels=None, label=None, title_prefix="Signal"):
    """
    Plot a short time span from a selected or random signal.
    
    Parameters
    ----------
    data_dict : dict[label] -> np.ndarray [samples, channels]
        Dictionary of signals.
    fs : int
        Sampling rate in Hz.
    seconds : float
        Duration of the plotted segment in seconds.
    channels : list[int] or None
        Which channels to plot (0-based). None means all.
    label : str or None
        If provided, plot this class label; if None, pick randomly.
    title_prefix : str
        Title prefix for the plot.
    """
    if not data_dict:
        raise ValueError("data_dict is empty")

    # Choose label
    if label is None:
        label = random.choice(list(data_dict.keys()))
    elif label not in data_dict:
        raise ValueError(f"Label '{label}' not found. Available: {list(data_dict.keys())[:5]}...")

    x = data_dict[label]
    if x.ndim == 1:
        x = x[:, None]

    n_show = min(int(seconds * fs), x.shape[0])
    if n_show < 1:
        raise ValueError("seconds too small for given fs")

    if channels is None:
        ch_idx = range(x.shape[1])
    else:
        ch_idx = [c for c in channels if 0 <= c < x.shape[1]]

    t = np.arange(n_show) / fs

    plt.figure(figsize=(12, 4))
    for c in ch_idx:
        plt.plot(t, x[:n_show, c], label=f"ch{c}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{title_prefix}: {label}  fs={fs/1000:.1f} kHz")
    plt.legend()
    plt.tight_layout()
    plt.show()

import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

# Rates
sampling_rate_cwru = 12000
target_sampling_rate_cwru = 25600
UPSAMPLE, DOWNSAMPLE = 32, 15

def compare_resampling(mat_path, channel="DE", max_len=6000):
    """
    Plot a short segment of the original and resampled signals
    from a given CWRU .mat file.
    """
     # --- Load original signal ---
    m = sio.loadmat(mat_path)
    key = next((k for k in m.keys() if not k.startswith('__') and k.endswith(f"_{channel}_time")), None)
    if key is None:
        raise ValueError(f"❌ Channel {channel} not found in {mat_path}")
    x = np.asarray(m[key]).squeeze().astype(np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    x = x[:max_len, None]  # [T,1]

    # --- Resample ---
    y = resample_poly(x, UPSAMPLE, DOWNSAMPLE, axis=0).astype(np.float32)

    # --- Time axes ---
    t_orig = np.arange(x.shape[0]) / sampling_rate_cwru
    t_resampled = np.arange(y.shape[0]) / target_sampling_rate_cwru

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(t_resampled, y[:, 0], label=f"Resampled {target_sampling_rate_cwru/1000:.1f} kHz", alpha=0.7)
    plt.plot(t_orig, x[:, 0], label=f"Original {sampling_rate_cwru/1000:.1f} kHz", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Resampling comparison: {os.path.basename(mat_path)} ({channel})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_signal_examples(data_dict, class_labels, sampling_rate, window_size):
    """
    Enhanced function to plot signal examples with proper feature extraction pipeline.
    
    This function demonstrates the complete signal processing pipeline used in extract_features:
    1. Signal segmentation
    2. Z-score normalization per segment
    3. Hilbert transform for vibration sensors (channels 0, 3)
    4. FFT transformation
    5. Comparison of raw vs processed signals
    """
    
    # Calculate frequency array for FFT plots
    freqs = np.fft.fftfreq(window_size, d=1/sampling_rate)[:window_size // 2]
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # Randomly select a class and sensor
    label = class_labels[np.random.randint(0, len(class_labels))]
    data = data_dict[label]
    num_samples, num_sensors = data.shape
    
    # Select a vibration sensor (0 or 3)
    sensor_idx = np.random.choice([0,1])
    sensor_data = data[:, sensor_idx]
    
    # Select a random segment
    num_segments = num_samples // window_size
    random_idx = np.random.randint(0, num_segments)
    segment = sensor_data[random_idx * window_size:(random_idx + 1) * window_size]
    
    # Apply the same processing pipeline as extract_features
    # 1. Z-score normalization
    segment_mean = np.mean(segment)
    segment_std = np.std(segment)
    if segment_std == 0:
        segment_std = 1e-8
    segment_normalized = (segment - segment_mean) / segment_std
    
    # 2. Hilbert transform (for vibration sensors)
    segment_hilbert = np.abs(hilbert(segment_normalized))
    
    # 3. FFT transformation
    fft_raw = np.abs(fft(segment))[:window_size // 2]
    fft_processed = np.abs(fft(segment_hilbert))[:window_size // 2]
    fft_processed[0] = 0  # Remove DC component for clarity"
    
    # Plotting
    # Raw Signal
    axes[0].plot(segment, label="Raw Signal", linewidth=1)
    axes[0].set_title(f"Raw Signal - {label} (Sensor {sensor_idx})")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Normalized Signal
    axes[1].plot(segment_normalized, label="Normalized", color='orange', linewidth=1)
    axes[1].set_title("Z-score Normalized Signal")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Hilbert Transform Envelope
    axes[2].plot(segment_hilbert, label="Hilbert Envelope", color='red', linewidth=1)
    axes[2].set_title("Hilbert Transform Envelope")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Envelope Amplitude")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # FFT Comparison
    axes[3].plot(freqs, fft_raw, label="Raw FFT", alpha=0.7, linewidth=1)
    axes[3].plot(freqs, fft_processed, label="Processed FFT", alpha=0.7, linewidth=1)
    axes[3].set_title("Frequency Spectrum Comparison")
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("Magnitude")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


