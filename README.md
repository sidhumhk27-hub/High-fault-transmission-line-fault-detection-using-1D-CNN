# Real-Time Fault Detection in High-Voltage Transmission Lines Using 1D-CNN

This repository contains an end-to-end deep learning pipeline designed to monitor, detect, and classify transient faults in high-voltage electrical transmission systems. Utilizing high-frequency current and voltage sensor data, the core architecture leverages a **1D Convolutional Neural Network (1D-CNN)** to deliver rapid, sub-cycle fault classification, ensuring grid stability and preventing catastrophic hardware failures.

## 📌 Project Overview

Faults in high-voltage transmission lines (such as short-circuits, overloads, and ground faults) must be isolated within milliseconds to protect grid infrastructure. Traditional distance relays can struggle with high-resistance faults or dynamic loading conditions.

This project addresses these challenges by processing raw multi-channel time-series data directly through a 1D-CNN, bypassing manual feature extraction to achieve highly accurate, real-time fault categorization and localization.

---

## 🚀 Key Features

* **High-Frequency Temporal Analysis:** Directly processes time-series current ($I$) and voltage ($V$) waveforms using 1D convolutional layers.
* **Comprehensive Preprocessing:** Scalable data pipeline handling normalization, missing value handling, and structural reshaping for sequence modeling.
* **Explainable AI (XAI):** Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization to pinpoint exactly which cycles or transient peaks triggered the model's fault classification.
* **Edge-Ready Deployment:** Export pipeline configured to optimize and convert trained models into highly compact **TensorFlow Lite (TFLite)** formats for low-latency deployment on microcontrollers and embedded edge relays.

---

## 🛠️ System Architecture & Workflow

The pipeline bridges conceptual grid telemetry with hardware-efficient inference models through the following sequence:

```
  [ Sensor Data ] ──> [ 1D-CNN Feature Extraction ] ──> [ Fault Classification ]
                                                                │
         ┌──────────────────────────────────────────────────────┴────────────────────────────────────────────────┐
         ▼                                                                                                       ▼
[ Grad-CAM Interpretability ]                                                                          [ TFLite Edge Deployment ]
(Visualizes transient fault triggers)                                                                  (Optimized for hardware relays)

```

1. **Data Ingestion & Scaling:** Raw inputs are standardized to ensure variance stability across varying line impedances.
2. **Feature Extraction:** 1D convolutions slide across the temporal axis to capture localized phase shifts and sudden current spikes.
3. **Classification Layer:** Dense layers output prediction probabilities across various fault profiles (e.g., Line-to-Line, Line-to-Ground, Triple Line).
4. **Model Diagnostics:** Evaluated using confusion matrices, precision-recall curves, and regional Grad-CAM overlays.

---

## 💻 Prerequisites & Setup

### Dependencies

Ensure you have Python 3.8+ installed along with the following primary libraries:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

```

### Running the Pipeline

To train the model, evaluate its performance, and export the edge-ready TFLite binaries, execute the core script:

```bash
python Fault_detection.py

```

---

## 📊 Model Interpretability & Edge Deployment

### Grad-CAM Localization

To prevent the deep learning model from operating as a complete "black box," the repository maps internal gradients back onto the input waveform. This highlights the precise millisecond window where abnormal phase deviations or surge currents initiated, offering power systems engineers auditable decision pathways.

### Embedded Optimization

The final stage of the pipeline automatically converts the saved Keras architecture into a serialized `.tflite` model. This enables:

* Significant reductions in memory footprint.
* Low-latency execution on localized embedded hardware right at the substation level, bypassing the need for cloud dependency.
