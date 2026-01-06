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

from collections import Counter

def summarize_features(X, y, class_labels, name="dataset"):
    """
    Quick integrity report for feature tensors.
    X: [n_segments, n_bins, n_channels]
    y: [n_segments] integer ids aligned with class_labels
    """
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
    assert X.ndim == 3, f"X must be 3D [segments, bins, channels], got {X.shape}"
    assert y.ndim == 1, f"y must be 1D [segments], got {y.shape}"
    assert X.shape[0] == y.shape[0], "Segments count mismatch between X and y"

    # NaN or Inf check
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()

    # Basic stats
    n_segments, n_bins, n_channels = X.shape
    value_min = float(np.nanmin(X))
    value_max = float(np.nanmax(X))
    value_mean = float(np.nanmean(X))

    # Label coverage
    cnt = Counter(y.tolist())
    label_rows = []
    for i, lbl in enumerate(class_labels):
        label_rows.append((lbl, cnt.get(i, 0)))

    print(f"\n[{name}] Summary")
    print(f"  Shape X: {X.shape}  y: {y.shape}")
    print(f"  Channels: {n_channels}  Bins: {n_bins}")
    print(f"  Stats  min:{value_min:.3f}  max:{value_max:.3f}  mean:{value_mean:.3f}")
    print(f"  NaN present: {has_nan}  Inf present: {has_inf}")
    print("  Samples per class:")
    for lbl, n in label_rows:
        print(f"   - {lbl}: {n}")

    # Simple per channel energy check
    energy = np.mean(np.sum(X**2, axis=1), axis=0)  # mean over segments of sum over bins per channel
    print("  Mean spectral energy per channel:", np.round(energy, 3))
    
def plot_feature_example(X, y, class_labels, fs, label=None, idx=None, channel=0):
    """
    Plot one feature vector: magnitude spectrum of a chosen segment and channel.
    If label is given chooses the first sample of that class. If idx is given uses that index.
    """
    n_segments, n_bins, n_channels = X.shape
    if channel >= n_channels:
        raise ValueError(f"channel {channel} out of range {n_channels}")

    if idx is None:
        if label is None:
            idx = 0
        else:
            cls_id = class_labels.index(label)
            where = np.where(y == cls_id)[0]
            if len(where) == 0:
                raise ValueError(f"No segments for label {label}")
            idx = int(where[0])
 
    freqs = np.fft.fftfreq(n_bins*2, d=1/fs)[:n_bins]  # consistent with window_size//2 bins
    spec = X[idx, 1:, channel]  # remove DC (first bin)
    freqs = freqs[1:]           # remove DC frequency

    plt.figure(figsize=(10,4))
    plt.plot(freqs, spec, lw=1)
    ttl_label = class_labels[y[idx]] if 0 <= y[idx] < len(class_labels) else f"id {y[idx]}"
    plt.title(f"Spectrum example  idx={idx}  label={ttl_label}  ch={channel}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_class_mean_spectra(X, y, class_labels, fs, channel=0):
    """
    Plot mean spectrum per class for a chosen channel for a quick separability check.
    """
    n_segments, n_bins, n_channels = X.shape
    if channel >= n_channels:
        raise ValueError(f"channel {channel} out of range {n_channels}")
    freqs = np.fft.fftfreq(n_bins*2, d=1/fs)[:n_bins]
    freqs = freqs[1:]  # remove DC frequency

    plt.figure(figsize=(12,6))
    for cid, lbl in enumerate(class_labels):
        idxs = np.where(y == cid)[0]
        if len(idxs) == 0:
            continue
        mean_spec = X[idxs, 1:, channel].mean(axis=0)
        plt.plot(freqs, mean_spec, lw=1, label=lbl)
    plt.title(f"Mean spectra per class  ch={channel}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
