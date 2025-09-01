###################################################################################
# Publisher.py, EDGE-TO-CLOUD FAULT DIAGNOSIS - PUBLISHER
# This script simulates the process of publishing vibration data to an MQTT broker.
# Created on: 2025-08-31
# Author: Joaquín López
##################################################################################

import pandas as pd
import time
import json
import os
import argparse
import random
import paho.mqtt.client as mqtt
from datetime import datetime

# Device to simulate: motor1 motor2 motor3
parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.getenv("DEVICE_ID", "motor1"))
args = parser.parse_args()
DEVICE_ID = args.device  # "motor1" | "motor2" | "motor3"
print(f"🔧 Simulating device: {DEVICE_ID}")

# Configuration selector
var = 1  # 0 for public, 1 for personal HiveMQ, 2 for personal EMQX

# MQTT Configuration
if var == 0:
    # Public HiveMQ broker (no TLS, no auth)
    MQTT_BROKER = "broker.hivemq.com"
    MQTT_PORT = 1883
    MQTT_USERNAME = None
    MQTT_PASSWORD = None
elif var == 1:
    # Personal HiveMQ broker
    #MQTT_BROKER = "d6343f2567d641e4a0e22d56e9492a04.s1.eu.hivemq.cloud"
    MQTT_BROKER = "5dcd5cd5264848d1a3c069eb39cf819e.s1.eu.hivemq.cloud"
    MQTT_PORT = 8883
    MQTT_USERNAME = "publisher"
    MQTT_PASSWORD = "joacoL21"
else:
    # Personal EMQX broker
    MQTT_BROKER = "zb191690.ala.eu-central-1.emqxsl.com"
    MQTT_PORT = 8883
    MQTT_USERNAME = "publisher"
    MQTT_PASSWORD = "joacoL21"

MQTT_TOPIC_STORAGE = f"Enterprise/Site/Area/{DEVICE_ID}/Edge/MotorModel/vibration"
MQTT_TOPIC_COMPUTE  = f"Enterprise/Site/Area/{DEVICE_ID}/Analysis/Vibration/raw_vector"

# MQTT Client Setup
client = mqtt.Client(client_id=f"vibration_data_publisher{DEVICE_ID}", clean_session=True)

if var == 1:
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
elif var == 2:
    client.tls_set(ca_certs="../Certificate/emqxsl-ca.crt")  # ← This is the path to the CA cert
    client.tls_insecure_set(False)  # ← Enforce certificate validation

client.connect(MQTT_BROKER, MQTT_PORT)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected successfully to MQTT broker!")
    else:
        print(f"❌ Connection failed with error code {rc}")

def on_disconnect(client, userdata, rc):
    print(f"⚠️ Disconnected from MQTT broker (rc={rc}).")
    try:
        print(rc)# client.reconnect()
    except Exception as e:
        print(f"❌ Reconnection failed: {e}")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Class labels and sampling info
class_labels = ['0Nm_Normal', '0Nm_BPFI_03', '0Nm_BPFO_03', '0Nm_Misalign_01', '0Nm_Unbalance_0583mg']
# Per device RNG so instances do not pick identical labels in lockstep
rng = random.Random(DEVICE_ID)
rng.shuffle(class_labels)

fs = 25_600
fs_store = 1024
downsample_factor = fs // fs_store
publish_period = 1  # seconds
# Calculate batch sizes based on the downsample factor
batch_size_compute = int(fs * publish_period)
batch_size_storage = int(fs_store * publish_period)

# Functions
def publish_compute_batch(batch, batch_id):
    if not client.is_connected():
        print("⚠️ MQTT client not connected. Skipping compute batch.")
        return
    payload = {
        "batch_id": batch_id,
        "data": batch.tolist()
    }
    json_str = json.dumps(payload)
    payload_size = round(len(json_str.encode('utf-8')) / 1024, 2)  # Convert to KB
    result = client.publish(MQTT_TOPIC_COMPUTE, json_str, qos=0)
    print(f"📡 Compute batch {batch_id} published with size {payload_size} KB, result: {result.rc}")

def publish_storage_batch(batch, batch_id, start_timestamp_ns):
    if not client.is_connected():
        print("⚠️ MQTT client not connected. Skipping storage batch.")
        return
    payload = {
        "batch_id": batch_id,
        "timestamp": start_timestamp_ns,
        "sensor_data": []
    }
    for i, row in enumerate(batch):
        row_timestamp = start_timestamp_ns + int(i * (1e9 / fs_store))
        payload["sensor_data"].append({
            "timestamp": row_timestamp,
            "sensor1": float(row[0]),
            "sensor2": float(row[1]),
            "sensor3": float(row[2]),
            "sensor4": float(row[3])
        })
    json_str = json.dumps(payload)
    payload_size = round(len(json_str.encode('utf-8')) / 1024, 2)  # Convert to KB
    result = client.publish(MQTT_TOPIC_STORAGE, json_str, qos=0)
    print(f"📡 Storage batch {batch_id} published with size {payload_size} KB, result: {result.rc}")

# Randomized simulation loop
print(f"\n🚀 Starting simulation for device: {DEVICE_ID}")
# small stagger to reduce sync across devices
time.sleep(rng.uniform(0.0, 1.0))


while True:
    class_label = rng.choice(class_labels)
    print(f"\n📚 Device {DEVICE_ID} will publish class: {class_label}")

    csv_path = f"./data/vibration{round(fs/1000, 2)}kHz/{class_label}.csv"
    try:
        df_compute = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_path}")
        time.sleep(1.0)
        continue

    df_storage = df_compute.iloc[::downsample_factor, :].reset_index(drop=True)

    total_samples_compute = df_compute.shape[0]
    total_samples_storage = df_storage.shape[0]
    index_compute = index_storage = 0
    batch_id = 0

    while index_compute < total_samples_compute and index_storage < total_samples_storage:
        print(f"📤 Publishing batch {batch_id} for class {class_label} from {DEVICE_ID}")

        batch_compute = df_compute.iloc[index_compute:index_compute + batch_size_compute, :].values
        publish_compute_batch(batch_compute, f"{class_label}:{batch_id}")
        index_compute += batch_size_compute

        batch_storage = df_storage.iloc[index_storage:index_storage + batch_size_storage, :].values
        start_timestamp_ns = int(time.time() * 1e9)
        publish_storage_batch(batch_storage, f"{class_label}:{batch_id}", start_timestamp_ns)
        index_storage += batch_size_storage

        print(f"✅ Batch {batch_id} published for {DEVICE_ID} at {datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}")
        batch_id += 1

        # keep original timing
        time.sleep(batch_size_compute / fs)