# NeuroIntent: Neuromorphic Stroke Rehabilitation Assistant

## Overview

NeuroIntent is a neuromorphic healthcare system that detects motor intent from EMG signals and demonstrates how Spiking Neural Networks (SNNs) can be used for low-power assistive rehabilitation technologies.

The system converts EMG signals into spike trains, processes them through a biologically inspired Spiking Neural Network, and predicts user motor intent that can be used to control assistive devices such as prosthetic limbs, rehabilitation robots, and exoskeletons.

---

## Problem Statement

Stroke patients often retain neural intent before physical movement becomes possible.

Traditional rehabilitation systems rely on observable movement, delaying assistance and limiting recovery opportunities.

NeuroIntent aims to bridge this gap by detecting motor intent directly from biosignals and enabling intelligent assistive responses.

---

## Key Features

* Real-time EMG signal processing
* Spike-based neuromorphic encoding
* Spiking Neural Network inference
* Motor intent classification
* Virtual prosthetic response simulation
* Interactive Streamlit dashboard
* Spike activity visualization
* Low-power neuromorphic computing framework

---

## System Architecture

EMG Signal
→ Feature Extraction
→ Spike Encoding
→ Spiking Neural Network
→ Motor Intent Detection
→ Assistive Device Response

---

## Technologies Used

### AI & Machine Learning

* PyTorch
* Scikit-learn
* NumPy

### Neuromorphic Computing

* Spiking Neural Networks
* Leaky Integrate-and-Fire Neurons
* Rate-Based Spike Encoding

### Application Layer

* Streamlit
* Matplotlib
* Pandas

---

## Dataset

The system was trained using approximately 16,000 EMG samples.

Dataset Characteristics:

* 64 extracted features
* 4 gesture classes
* Motor intent classification task

---

## Results
[Click Here](https://neurointent-ggdcjyzrbcjgzx7cj97wwk.streamlit.app/)

### ANN

* Accuracy: 56%

### SNN

* Accuracy: 24.18%

Although the ANN currently achieves higher accuracy, the SNN demonstrates the key neuromorphic advantages of event-driven processing, sparse computation, and potential energy efficiency.

---

## Future Work

* EEG + EMG multimodal fusion
* Real-time hardware integration
* Loihi neuromorphic deployment
* Adaptive online learning
* Prosthetic hand integration
* Exoskeleton control systems

---

## Hackathon Submission

This project was developed as a prototype for neuromorphic stroke rehabilitation and demonstrates the feasibility of low-power motor intent detection using brain-inspired computing.

---

## Team

Synapse Squad

M.Tech Bioinformatics and Artificial Intelligence

---
PROJECT DEMONSTRATION VIDEO
[CLick Here](https://github.com/user-attachments/assets/43d741da-b812-40e9-9355-8996db49bd93)

