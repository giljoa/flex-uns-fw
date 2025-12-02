import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import json
import time
import os
import re
import argparse

from tensorflow import keras
from scipy.signal import hilbert
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler

# Load CNN Model
model = keras.models.load_model('./ml/models/multi_domain/multi_domain_model_fold_1.h5')

# Configuration selector
var = 1  # 0 public, 1 private

# MQTT Configuration
if var == 0:
    MQTT_BROKER = "broker.hivemq.com"
    MQTT_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
else:
    MQTT_BROKER = "d6343f2567d641e4a0e22d56e9492a04.s1.eu.hivemq.cloud"
    #MQTT_BROKER = "5dcd5cd5264848d1a3c069eb39cf819e.s1.eu.hivemq.cloud"
    MQTT_PORT = 8883
    MQTT_USERNAME = "diagnosis"
    MQTT_PASSWORD = "joacoL21"

# Optional device filter
parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.getenv("DEVICE_ID", ""))
args = parser.parse_args()
DEVICE_FILTER = args.device.strip()

BASE = "Enterprise/Site/Area"
if DEVICE_FILTER:
    MQTT_TOPIC_RAW = f"{BASE}/{DEVICE_FILTER}/Analysis/Vibration/raw_vector"
else:
    MQTT_TOPIC_RAW = f"{BASE}/+/Analysis/Vibration/raw_vector"

# Initialize MQTT Client
client = mqtt.Client(client_id=f"vibration_diagnosis_{DEVICE_FILTER or 'all'}", clean_session=True)

if var == 1:
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Parameters
window_size = 5120
class_labels = ['0Nm_Normal', '0Nm_BPFI_03', '0Nm_BPFO_03', '0Nm_Misalign_01', '0Nm_Unbalance_0583mg']

def on_message(client, userdata, message):
    try:
        # Extract device id from topic
        topic = message.topic
        m = re.match(rf"^{BASE}/([^/]+)/Analysis/Vibration/raw_vector$", topic)
        device_id = m.group(1) if m else "unknown"

        # Decode and parse JSON
        payload = json.loads(message.payload.decode("utf-8"))
        batch_data = np.array(payload["data"])  # Shape: (N, num_sensors)
        batch_id = payload.get("batch_id", "Unknown")
        timestamp = payload.get("timestamp", int(time.time() * 1e9))

        if batch_data.shape[0] < window_size:
            print(f"Not enough samples in batch {batch_id}. Got {batch_data.shape[0]} need {window_size}.")
            return

        print(f"Batch {batch_id} from {device_id} received. Processing...")

        # Extract last window_size samples
        data = batch_data[-window_size:, :]
        num_sensors = data.shape[1]

        # Feature extraction
        X = []
        fft_data = []
        num_fft_bins = window_size // 2
        fft_timestamps = [timestamp] * num_fft_bins
        fft_transformed = np.zeros((num_sensors, num_fft_bins))

        for sensor_idx in range(num_sensors):
            sensor_data = data[:, sensor_idx]
            sensor_data = StandardScaler().fit_transform(sensor_data.reshape(-1, 1)).flatten()
            if sensor_idx in [0, 3]:
                sensor_data = np.abs(hilbert(sensor_data))
            fft_features = np.abs(fft(sensor_data))[:num_fft_bins]
            X.append(fft_features)
            fft_transformed[sensor_idx, :] = fft_features

        for i in range(num_fft_bins):
            fft_entry = {"timestamp": int(fft_timestamps[i])}
            for sensor_idx in range(num_sensors):
                fft_entry[f"sensor{sensor_idx + 1}"] = float(fft_transformed[sensor_idx, i])
            fft_data.append(fft_entry)

        fft_payload = json.dumps({
            "batch_id": batch_id,
            "timestamp": timestamp,
            "fft_data": fft_data
        })

        MQTT_TOPIC_FFT_DEV = f"{BASE}/{device_id}/Analysis/Vibration/fft"
        client.publish(MQTT_TOPIC_FFT_DEV, fft_payload)
        print(f"FFT for batch {batch_id} published to {MQTT_TOPIC_FFT_DEV}")

        # Predict
        X_test = np.array(X)  # (num_sensors, num_fft_bins)
        Y_pred = model.predict(X_test, verbose=0)
        Y_pred = np.argmax(Y_pred, axis=1)
        final_prediction = class_labels[Y_pred[0]]
        fault_code = class_labels.index(final_prediction)

        json_payload = json.dumps({
            "prediction": final_prediction,
            "fault_code": fault_code,
            "timestamp": int(time.time() * 1e9),
            "device": device_id,
            "batch_id": batch_id
        })

        MQTT_TOPIC_PREDICT_DEV = f"{BASE}/{device_id}/Analysis/Diagnosis/prediction"
        client.publish(MQTT_TOPIC_PREDICT_DEV, json_payload)
        print(f"Prediction for batch {batch_id} published to {MQTT_TOPIC_PREDICT_DEV}")
        print("-----------------------------------------")

    except Exception as e:
        print(f"Error processing batch: {e}")

client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_RAW)

print(f"📡 Listening on '{MQTT_TOPIC_RAW}'...")
client.loop_start()

# Run service
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    client.loop_stop()
    client.disconnect()
    print("Diagnosis service stopped.")
