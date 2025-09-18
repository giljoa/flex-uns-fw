import matplotlib.pyplot as plt
import numpy as np
import random

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


def compare_resampling(mat_path, channel="DE", max_len=6000):
    """
    Plot a short segment of the original and resampled signals
    from a given CWRU .mat file.
    """
    # Load raw signal at 12 kHz
    x = _load_mat_signal(mat_path, max_len=max_len, channels=(channel,))
    # Resample to 25.6 kHz
    y = _resample_to_target(x, UPSAMPLE, DOWNSAMPLE)

    # Time axes
    t_orig = np.arange(x.shape[0]) / sampling_rate_cwru
    t_resampled = np.arange(y.shape[0]) / target_sampling_rate_cwru

    plt.figure(figsize=(12, 5))
    plt.plot(t_resampled, y[:,0], label=f"Resampled {target_sampling_rate_cwru/1000:.1f} kHz", alpha=0.7)
    plt.plot(t_orig, x[:,0], label=f"Original {sampling_rate_cwru/1000:.1f} kHz", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Resampling comparison for {os.path.basename(mat_path)} ({channel})")
    plt.legend()
    plt.tight_layout()
    plt.show()