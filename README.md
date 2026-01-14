# A Flexible Edge-to-Cloud Framework for Industrial Motor Fault Diagnosis with Unified Namespace Integration

## Overview  
This repository contains the implementation of a **Unified Namespace (UNS)-based IIoT framework** for **fault diagnosis in induction motors**. The framework is designed around three foundational pillars: **scalability, generalizability, and accessibility**.  

- **Scalability** is achieved through a **publisher script** that simulates multiple edge devices transmitting vibration data to an MQTT broker, and a **diagnosis script** that processes signals and publishes back motor state predictions.  
- **Generalizability** is validated with a **convolutional neural network (CNN)** trained on two benchmark datasets (KAIST and CWRU), demonstrating transfer learning for cross-dataset classification of common motor faults.  
- **Accessibility** is ensured by using **open-source tools** (Python, MQTT, InfluxDB, Grafana, Google Colab) and by emphasizing the **availability of industrial data** across all layers of the architecture.  

The result is a modular, interoperable framework that can be deployed in both academic and industrial contexts, avoiding vendor lock-in while supporting predictive maintenance strategies.

---

## Architecture  

The proposed system follows an **edge-to-cloud data flow** mediated by a **Unified Namespace (UNS):**

- **Edge Layer**: Python scripts (`Publisher.py`) replay vibration signals and publish raw and downsampled data.  
- **UNS**: MQTT hierarchical topics provide semantic context for all industrial data.  
- **Data Cloud**: Preprocessing (Hilbert + FFT) and CNN-based inference (`Diagnosis.py`).  
- **Historian**: InfluxDB stores raw and processed time-series data.  
- **Client Layer**: Grafana dashboards visualize real-time signals, FFT features, and predictions.  

---

## Datasets  
The framework supports **multi-domain fault diagnosis** using open-source datasets:

- **KAIST Dataset**: Vibration data from controlled experiments including normal, BPFI, BPFO, misalignment, and unbalance conditions under multiple load domains.  
- **CWRU Dataset**: Widely used bearing fault dataset from Case Western Reserve University, including inner race, outer race, and ball faults under varying loads and speeds.  

The CNN model is trained to classify faults across both datasets using **transfer learning**, demonstrating adaptability to heterogeneous data sources.

---

## Key Components  

- **Publisher (`Publisher.py`)**  
  Simulates multiple edge devices (motors) streaming vibration signals via MQTT.  

- **Diagnosis (`Diagnosis.py`)**  
  Subscribes to UNS topics, extracts FFT-based features, and classifies motor condition using a CNN model. Publishes diagnostic results back to the UNS.  

- **Model (`ml/models/multi_domain_model_fold_1.h5`)**  
  A 1D CNN trained for fault classification across KAIST and CWRU domains.  

- **Data Preparation (`data.ipynb`)**  
  Preprocessing pipeline including segmentation, normalization, Hilbert transform, and FFT.  

- **Flex - TIG Stack (`docker-compose.yaml`)**  
  Deployment of **Telegraf, InfluxDB, and Grafana** for persistent data storage and visualization.  


---

## Core Pillars  

- **Scalability**  
  The UNS decouples producers and consumers, enabling seamless expansion of devices, diagnostic models, or visualization clients.  

- **Generalizability**  
  Transfer learning ensures that the CNN model can classify faults from multiple datasets without structural modifications.  

- **Accessibility**  
  The framework leverages open-source tools and lightweight deployment requirements, allowing operation on low-cost devices and scaling to enterprise infrastructures.  

---

## Getting Started  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-repo>.git
   cd <your-repo>
   ```
2. Install dependencies
   ``bash
   pip install -r requirements.txt
   ```
3. Run the publisher simulation:
   ``bash
   python Publisher.py --device Motor1
   ```
4. Start the diagnosis service:
   ``bash
   python Diagnosis.py --device Motor1
   ```

## Results

The framework has been validated by:

- Running simulations with multiple edge devices to prove horizontal scalability.
- Integrating new datasets to demonstrate generalizability.
- Deploying exclusively with open-source and free tools to confirm accessibility.

Real-time dashboards provide end-to-end visibility from raw vibration signals to diagnostic predictions.

## Citation

If you use this repository in academic work, please cite:

- Gilbert Delgado. An IIoT Cloud Based Solution for Fault Diagnosis of Induction Motors. MSc Thesis, Universidad Industrial de Santander, 2025.
- Joaquín D. López et al. A Flexible Edge-to-Cloud Framework for Industrial Motor Fault Diagnosis with Unified Namespace Integration. (Submitted for journal publication).
