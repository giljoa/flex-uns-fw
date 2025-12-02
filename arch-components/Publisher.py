###################################################################################
# Publisher.py, EDGE TO CLOUD FAULT DIAGNOSIS - PUBLISHER
# Randomly publishes CSV batches from a chosen dataset folder:
#   ./data/publish-data/KAIST  or  ./data/publish-data/CWRU
###################################################################################

import pandas as pd
import numpy as np
import time
import math
import json
import os
import argparse
import random
import paho.mqtt.client as mqtt
from datetime import datetime

# ------------------------ CLI ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.getenv("DEVICE_ID", "motor1"))
parser.add_argument(
    "--dataset",
    choices=["KAIST", "CWRU"],
    default=os.getenv("DATASET", "KAIST"),
    help="Choose which dataset folder to stream from"
)
args = parser.parse_args()
DEVICE_ID = args.device
DATASET = args.dataset
print("Simulating device:", DEVICE_ID, " dataset:", DATASET)

# ------------------------ MQTT configuration ------------------------
var = 1  # 0 public, 1 personal HiveMQ, 2 personal EMQX

if var == 0:
    MQTT_BROKER = "broker.hivemq.com"
    MQTT_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
elif var == 1:
    MQTT_BROKER = "d6343f2567d641e4a0e22d56e9492a04.s1.eu.hivemq.cloud"
    #MQTT_BROKER = "5dcd5cd5264848d1a3c069eb39cf819e.s1.eu.hivemq.cloud"
    MQTT_PORT = 8883
    MQTT_USERNAME = "publisher"
    MQTT_PASSWORD = "joacoL21"
else:
    MQTT_BROKER = "zb191690.ala.eu-central-1.emqxsl.com"
    MQTT_PORT = 8883
    MQTT_USERNAME = "publisher"
    MQTT_PASSWORD = "joacoL21"

MQTT_TOPIC_STORAGE = F"Enterprise/Site/Area/{DEVICE_ID}/Edge/MotorModel/vibration"
MQTT_TOPIC_COMPUTE = F"Enterprise/Site/Area/{DEVICE_ID}/Analysis/Vibration/raw_vector"

client = mqtt.Client(client_id=F"vibration_data_publisher_{DEVICE_ID}", clean_session=True)
if var == 1:
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
elif var == 2:
    client.tls_set(ca_certs="../Certificate/emqxsl-ca.crt")
    client.tls_insecure_set(False)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT")
    else:
        print("Connection failed rc={}".format(rc))

def on_disconnect(client, userdata, rc):
    print("Disconnected rc={}".format(rc))

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# ------------------------ Sampling and batching ------------------------
fs = 25_600
fs_store = 1024
downsample_factor = fs // fs_store
publish_period = 1
batch_size_compute = int(fs * publish_period)
batch_size_storage = int(fs_store * publish_period)

# ------------------------ Resolve dataset folder and files ------------------------
DATA_DIR = os.path.join(".", "data", "publish-data", DATASET)
if not os.path.isdir(DATA_DIR):
    raise RuntimeError(F"Dataset folder not found: {DATA_DIR}")

csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
if not csv_files:
    raise RuntimeError(F"No CSV files found in {DATA_DIR}")

# Per device RNG so multiple instances avoid lockstep
rng = random.Random(str(DEVICE_ID) + str(time.time_ns()))

# ------------------------ Helpers ------------------------
def publish_compute_batch(batch, batch_id):
    if not client.is_connected():
        print("MQTT not connected. Skip compute batch.")
        return
    payload = {
        "batch_id": batch_id,
        "data": batch.tolist()
    }
    js = json.dumps(payload)
    result = client.publish(MQTT_TOPIC_COMPUTE, js, qos=0)
    print("Compute batch {} rc={}".format(batch_id, result.rc))

def publish_storage_batch(batch, batch_id, start_timestamp_ns):
    if not client.is_connected():
        print("MQTT not connected. Skip storage batch.")
        return

    # Build sensor_data with dynamic channel count
    n_cols = batch.shape[1]
    sensor_data = []
    step_ns = int(1e9 / fs_store)
    for i, row in enumerate(batch):
        entry = {"timestamp": start_timestamp_ns + i * step_ns}
        for c in range(n_cols):
            entry[f"sensor{format(c + 1)}"] = float(row[c])
        sensor_data.append(entry)

    payload = {
        "batch_id": batch_id,
        "timestamp": start_timestamp_ns,
        "sensor_data": sensor_data
    }
    js = json.dumps(payload)
    result = client.publish(MQTT_TOPIC_STORAGE, js, qos=0)
    print("Storage batch {} rc={}".format(batch_id, result.rc))

# ------------------------ Main loop ------------------------
print("Starting simulation in folder:", DATA_DIR)

# Wait up to 3 seconds for MQTT connection
t0 = time.time()
while not client.is_connected() and time.time() - t0 < 3.0:
    time.sleep(0.05)
    
while True:
    csv_path = rng.choice(csv_files)
    class_label = os.path.splitext(os.path.basename(csv_path))[0]
    print("Chosen file:", class_label + ".csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error reading {}: {}".format(csv_path, e))
        time.sleep(1.0)
        continue

    # Ensure a numpy array with shape [N, C]
    arr = df.values
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Validate channel count per dataset
    n_cols = arr.shape[1]
    if DATASET == "KAIST":
        if n_cols != 4:
            print(f"KAIST csv must have 4 columns. Found {n_cols}. Skip.")
            time.sleep(1.0)
            continue
    else:
        if n_cols not in (2, 3):
            print(f"CWRU csv must have 2 or 3 columns. Found {n_cols}. Skip.")
            time.sleep(1.0)
            continue

    # Repeat the file 3 times
    arr = np.tile(arr, (3, 1))

    # Downsample for storage topic
    arr_store = arr[::downsample_factor, :]

    total_compute = arr.shape[0]
    total_store = arr_store.shape[0]

    n_batches_compute = int(math.ceil(float(total_compute) / float(batch_size_compute)))
    n_batches_store  = int(math.ceil(float(total_store) / float(batch_size_storage)))
    n_batches = max(n_batches_compute, n_batches_store)

    for batch_id in range(n_batches):
        # Compute slice
        start_c = batch_id * batch_size_compute
        end_c   = min(start_c + batch_size_compute, total_compute)
        if start_c < end_c:
            batch_compute = arr[start_c:end_c, :]
            publish_compute_batch(batch_compute, f"{class_label}:{batch_id}")

        # Storage slice
        start_s = batch_id * batch_size_storage
        end_s   = min(start_s + batch_size_storage, total_store)
        if start_s < end_s:
            start_ts_ns = int(time.time() * 1e9)
            batch_storage = arr_store[start_s:end_s, :]
            publish_storage_batch(batch_storage, f"{class_label}:{batch_id}", start_ts_ns)

        print(f"---Published batch {batch_id} at {datetime.now().strftime('%H:%M:%S')}---")

        # Keep one second pacing aligned to compute sampling
        time.sleep(batch_size_compute / fs)

