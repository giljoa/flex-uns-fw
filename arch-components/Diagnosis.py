###################################################################################
# Diagnosis.py, EDGE TO CLOUD FAULT DIAGNOSIS - Diagnosis
# Runs ML transfer model to diagnose vibration signals from the broker
# and publishes back the results
###################################################################################
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import json
import time
import os
import re
import argparse
import ntplib

from tensorflow import keras
from scipy.signal import hilbert
from scipy.fftpack import fft

from datetime import datetime, timezone
from collections import defaultdict

clock_offsets_ns = defaultdict(list)

# --------------- Load CNN Model ---------------
model = keras.models.load_model('/content/drive/MyDrive/FlexUNS/kaist_to_cwru_transfer.keras')

# --------------- Configuration selector ---------------
var = 1  # 0 public, 1 private

# --------------- NTP Server helper ---------------
# (Used when not on cloud)
NTP_SERVER = "pool.ntp.org"
SYNC_INTERVAL_SEC = 60

_ntp_offset_ns = 0
_last_sync_sec = 0.0

def _sync_ntp_if_needed():
    global _ntp_offset_ns, _last_sync_sec
    now_sec = time.time()
    if now_sec - _last_sync_sec < SYNC_INTERVAL_SEC:
        return
    try:
        c = ntplib.NTPClient()
        resp = c.request(NTP_SERVER, version=3)
        ntp_ns = int(resp.tx_time * 1e9)
        local_ns = time.time_ns()
        _ntp_offset_ns = ntp_ns - local_ns
        _last_sync_sec = now_sec
        print(f"NTP sync ok offset_ns={_ntp_offset_ns}")
    except Exception as e:
        print(f"NTP sync failed {e}")

def now_ntp_ns():
    _sync_ntp_if_needed()
    return time.time_ns() + _ntp_offset_ns

# --------------- MQTT Configuration ---------------
if var == 0:
    MQTT_BROKER = "broker.hivemq.com"
    MQTT_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
else:
    #MQTT_BROKER = "d6343f2567d641e4a0e22d56e9492a04.s1.eu.hivemq.cloud"
    MQTT_BROKER = "4e636e5bce054be2a6aa2a51659f12ed.s1.eu.hivemq.cloud"
    MQTT_PORT = 8883
    MQTT_USERNAME = "diagnosis"
    MQTT_PASSWORD = "joacoL21"

# --------------- Optional device filter ---------------
# (If running on local python venv to use arguments)
# parser = argparse.ArgumentParser()
# parser.add_argument("--device", default=os.getenv("DEVICE_ID", ""))
# args = parser.parse_args()
# DEVICE_FILTER = args.device.strip()
DEVICE_FILTER = 0

BASE = "Enterprise/Site/Area"
if DEVICE_FILTER:
    MQTT_TOPIC_RAW = f"{BASE}/{DEVICE_FILTER}/Analysis/Vibration/raw_vector"
else:
    MQTT_TOPIC_RAW = f"{BASE}/+/Analysis/Vibration/raw_vector"

# --------------- MQTT Client ---------------
client = mqtt.Client(client_id=f"vibration_diagnosis_{DEVICE_FILTER or 'all'}", clean_session=True)
if var == 1:
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# --------------- Parameters ---------------
window_size = 5120
num_fft_bins = window_size // 2
# Use a single fixed channel for inference to keep a consistent input with training
used_channels = [0]
hilbert_channels = (0, 3)
class_labels = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance', 'Ball']

# --------------- Feature extraction helpers ---------------
def zscore_per_channel(seg_2d):
    """
    seg_2d shape [win, C]
    returns z scored array with same shape
    """
    mu = seg_2d.mean(axis=0, keepdims=True)
    sd = seg_2d.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1e-8, sd)
    return (seg_2d - mu) / sd

def extract_fft_online(data_win, used_ch, hilb_ch):
    """
    data_win shape [win, C_full]
    Select channels, z score, apply Hilbert on selected channels if present,
    compute magnitude FFT over positive half. Returns spec shape [bins, C_used].
    """
    # Enforce consistent channels
    c_full = data_win.shape[1]
    if any(ch >= c_full for ch in used_ch):
        raise ValueError("Requested channel index out of range")
    seg = data_win[:, used_ch]  # [win, C_used]

    # Z score
    seg_z = zscore_per_channel(seg)

    # Hilbert envelope on indicated channels, but only if the index exists within used channels
    seg_proc = seg_z.copy()
    for hch in hilb_ch:
        if 0 <= hch < seg_proc.shape[1]:
            seg_proc[:, hch] = np.abs(hilbert(seg_proc[:, hch]))

    # FFT magnitude positive half per channel
    spec = np.abs(fft(seg_proc, axis=0))[: seg_proc.shape[0] // 2, :]  # [bins, C_used]
    return spec.astype(np.float32)

def on_message(client, userdata, message):
    try:
        # Parse device id from topic
        topic = message.topic
        m = re.match(rf"^{BASE}/([^/]+)/Analysis/Vibration/raw_vector$", topic)
        device_id = m.group(1) if m else "unknown"

        # Decode payload
        payload = json.loads(message.payload.decode("utf-8"))
        batch = np.array(payload["data"])
        batch_id = payload.get("batch_id", "Unknown")
        timestamp = payload.get("timestamp", "Unknown")

        ts_edge_ns = int(payload["timestamp"])
        ts_cloud_ns = time.time_ns() # use now_ntp_ns() if you are not on cloud

        raw_latency_ms = (ts_cloud_ns - ts_edge_ns) / 1e6

        edge_time_utc = datetime.fromtimestamp(ts_edge_ns / 1e9, tz=timezone.utc)
        cloud_time_utc = datetime.fromtimestamp(ts_cloud_ns / 1e9, tz=timezone.utc)

        print(f"[{device_id}] raw_latency_ms={raw_latency_ms:.3f}")
        print(f"[{device_id}] EDGE UTC : {edge_time_utc.isoformat()}")
        print(f"[{device_id}] CLOUD UTC: {cloud_time_utc.isoformat()}")
        print("-------------")

        if batch.shape[0] < window_size:
            print(f"Not enough samples in batch {batch_id}. Got {batch.shape[0]} need {window_size}.")
            return

        print(f"Batch {batch_id} from {device_id} received. Processing...")

        # Take the last window
        data = batch[-window_size:, :]  # [win, C]
        num_sensors = data.shape[1]

        # -------- FFT for visualization across all sensors (unchanged behavior) --------
        fft_transformed = np.zeros((num_sensors, num_fft_bins), dtype=np.float32)
        # Per sensor z score and optional Hilbert if sensor index matches
        for s_idx in range(num_sensors):
            ch_seg = data[:, s_idx:s_idx + 1]  # [win, 1]
            ch_z = zscore_per_channel(ch_seg)
            if s_idx in hilbert_channels:
                ch_z[:, 0] = np.abs(hilbert(ch_z[:, 0]))
            ch_spec = np.abs(fft(ch_z[:, 0]))[:num_fft_bins]
            fft_transformed[s_idx, :] = ch_spec.astype(np.float32)

        # Build per bin records with timestamp
        fft_timestamps = [timestamp] * num_fft_bins
        fft_data = []
        for i in range(num_fft_bins):
            row = {"timestamp": int(fft_timestamps[i])}
            for s_idx in range(num_sensors):
                row[f"sensor{s_idx + 1}"] = float(fft_transformed[s_idx, i])
            fft_data.append(row)

        fft_payload_dict = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "fft_data": fft_data
        }
        tmp_str = json.dumps(fft_payload_dict)
        size_mb = len(tmp_str.encode("utf-8")) / (1024 * 1024)
        fft_payload_dict["payload_size_mb"] = size_mb
        fft_payload = json.dumps(fft_payload_dict)

        MQTT_TOPIC_FFT_DEV = f"{BASE}/{device_id}/Analysis/Vibration/fft"
        client.publish(MQTT_TOPIC_FFT_DEV, fft_payload) # comment to save data exchange
        print(f"FFT for batch {batch_id} published to {MQTT_TOPIC_FFT_DEV}")

        # -------- Inference using the new consistent channel feature extraction --------
        spec_used = extract_fft_online(
            data_win=data,
            used_ch=used_channels,
            hilb_ch=hilbert_channels
        )  # [bins, C_used]

        # Shape to model input
        # Many Keras one dimensional CNNs expect [N, bins, C]
        X_infer = spec_used[np.newaxis, ...]  # [1, bins, C_used]

        # Predict
        logits = model.predict(X_infer, verbose=0)
        pred_idx = int(np.argmax(logits, axis=1)[0])

        prediction_dict = {
            "prediction": class_labels[pred_idx],
            "fault_code": pred_idx,
            "timestamp": int(time.time() * 1e9),
            "device": device_id,
            "batch_id": batch_id,
            "edge_to_cloud_latency_ms": raw_latency_ms
        }

        size_mb = len(json.dumps(prediction_dict).encode("utf-8")) / (1024 * 1024)
        prediction_dict["payload_size_mb"] = size_mb
        MQTT_TOPIC_METRICS = f"{BASE}/{device_id}/Metrics/cloud_diagnosis"
        client.publish(MQTT_TOPIC_METRICS, "%.2f" % size_mb)

        json_payload = json.dumps(prediction_dict)

        MQTT_TOPIC_PREDICT_DEV = f"{BASE}/{device_id}/Analysis/Diagnosis/prediction"
        client.publish(MQTT_TOPIC_PREDICT_DEV, json_payload)

        print(f"Prediction for batch {batch_id} published to {MQTT_TOPIC_PREDICT_DEV}")
        print("-----------------------------------------")

    except Exception as e:
        print(f"Error processing batch: {e}")

# --------------- Run ---------------
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_RAW)

print(f"📡 Listening on '{MQTT_TOPIC_RAW}'...")
client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    client.loop_stop()
    client.disconnect()
    print("Diagnosis service stopped.")