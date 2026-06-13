import streamlit as st
import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import sys
import time
# ==================================
# Import SNN Model
# ==================================

sys.path.append("src")

from model import SNN

# ==================================
# Page Configuration
# ==================================

st.set_page_config(
    page_title="NeuroIntent",
    layout="wide"
)

# ==================================
# Header
# ==================================

st.title(
    "🧠 NeuroIntent: Neuromorphic Stroke Rehabilitation Assistant"
)

st.markdown(
"""
### Real-Time Motor Intent Detection using Neuromorphic Computing

This system demonstrates how EMG signals can be converted into spikes and processed by a Spiking Neural Network (SNN) to detect motor intent for assistive rehabilitation devices.
"""
)

# ==================================
# Problem Statement
# ==================================

st.info(
"""
Stroke patients often retain neural intent before physical movement becomes possible.

NeuroIntent detects that intent using neuromorphic AI and can eventually trigger assistive prosthetics, exoskeletons, or rehabilitation devices in real time.
"""
)

# ==================================
# Dataset Statistics
# ==================================

data = pd.read_csv("data/spikes.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Samples",
        len(data)
    )

with col2:
    st.metric(
        "Features",
        data.shape[1]
    )

with col3:
    st.metric(
        "Classes",
        "4"
    )

# ==================================
# Model Performance
# ==================================

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "SNN Accuracy",
        "24.18%"
    )

with col2:
    st.metric(
        "ANN Accuracy",
        "56.00%"
    )

st.info(
"""
The ANN achieved higher classification accuracy on the current dataset,
while the SNN demonstrates a neuromorphic event-driven processing pipeline
designed for low-power intelligent assistive systems.
"""
)

# ==================================
# ANN vs SNN Comparison
# ==================================

st.subheader("🧠 ANN vs SNN")

comparison = pd.DataFrame(
    {
        "ANN": [
            "56.00% Accuracy",
            "Continuous Processing",
            "Higher Energy Usage",
            "Dense Computation"
        ],
        "SNN": [
            "24.18% Accuracy",
            "Event Driven",
            "Low Power Potential",
            "Sparse Spike Processing"
        ]
    }
)

fig, ax = plt.subplots()
ax.bar(["ANN", "SNN"], [56.00, 24.18], color=["#4CAF50", "#2196F3"])
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("ANN vs SNN Accuracy")
st.pyplot(fig)
# ==================================
# Pipeline
# ==================================

st.subheader("⚙️ Neuromorphic Processing Pipeline")

st.graphviz_chart("""digraph {EMG -> "Spike Encoding" "Spike Encoding" -> SNN SNN -> "Intent Detection" "Intent Detection" -> Prosthetic}""")

# ==================================
# Gesture Labels
# ==================================

gesture_map = {
    0: "Open Hand",
    1: "Close Hand",
    2: "Pinch",
    3: "Rest"
}

# ==================================
# Load Model
# ==================================

model = SNN()

model.load_state_dict(
    torch.load(
        "models/snn_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ==================================
# Mode Selection
# ==================================

st.subheader("🎮 Demo Controls")

mode = st.radio(
    "Select Mode",
    [
        "SNN Prediction",
        "Demo Mode"
    ]
)

# ==================================
# Prediction Button
# ==================================
synops = (64 * 128) + (128 * 4)
if st.button("🚀 Predict Gesture"):

    # --------------------------------
    # Real SNN Prediction
    # --------------------------------
    latency_ms = 0  
    if mode == "SNN Prediction":

        random_index = random.randint(
            0,
            len(data) - 1
        )

        sample = data.iloc[random_index].values

        sample_tensor = torch.tensor(
            sample,
            dtype=torch.float32
        ).unsqueeze(0)
        
        start_time = time.time()

        with torch.no_grad():

            output = model(sample_tensor)

            probabilities = F.softmax( output,dim=1)

            confidence = (torch.max(probabilities).item()* 100)

            prediction = torch.argmax(output,dim=1).item()
        
        latency_ms = (time.time() - start_time) * 1000

        st.write( f"Sample Index: {random_index}")

    # --------------------------------
    # Demo Mode
    # --------------------------------

    else:

        prediction = random.randint(0,3)

        confidence = random.uniform(82,98)

        latency_ms = random.uniform(0.5, 2.5)

        sample = data.iloc[ random.randint(0,len(data)-1)].values

    # ==================================
    # Confidence
    # ==================================

    st.info( f"Prediction Confidence: {confidence:.2f}%" )

    st.metric("Inference Latency",f"{latency_ms:.2f} ms")

    spike_rate = float(sample.mean())  # fraction of active spikes
    active_synops = int(synops * spike_rate)
    st.metric("Active SynOps (sparse)", f"{active_synops:,}")

    st.progress(  min(confidence / 100, 1.0) )

    # ==================================
    # Prediction Result
    # ==================================

    st.success(f"Detected Gesture: {gesture_map[prediction]}")

    # ==================================
    # Virtual Prosthetic Control Simulation
    # ==================================

    st.subheader( "🦾 Virtual Prosthetic Response")
    status = st.empty() 
    status.info( "Receiving Neuromorphic Signal..." ) 
    progress = st.progress(0) 
    for i in range(25): 
        progress.progress(i + 1) 
    status.info( "Decoding Motor Intent..." ) 
    for i in range(25, 50): 
        progress.progress(i + 1) 
    status.info( "Generating Prosthetic Command..." ) 
    for i in range(50, 75): 
        progress.progress(i + 1) 
    status.info( "Activating Assistive Device..." ) 
    for i in range(75, 100): 
        progress.progress(i + 1)

    if prediction == 0:

        st.write("🖐 OPEN HAND")

    elif prediction == 1:

        st.write("✊ CLOSED HAND")

    elif prediction == 2:

        st.write("🤏 PINCH")

    else:

        st.write("✋ REST")

    # ==================================
    # Spike Visualization
    # ==================================

    st.subheader(
        "⚡ Spike Activity Visualization"
    )

    fig, ax = plt.subplots(
        figsize=(10, 2)
    )

    ax.imshow(
        sample.reshape(1, -1),
        aspect="auto"
    )

    ax.set_xlabel(
        "EMG Features"
    )

    ax.set_yticks([])

    st.pyplot(fig)

# ==================================
# Footer
# ==================================

st.markdown("---")

st.caption(
"""
NeuroIntent | Neuromorphic Motor Intent Detection for Stroke Rehabilitation

Built using:
PyTorch • Norse • Streamlit • EMG Signals • Spiking Neural Networks
"""
)
