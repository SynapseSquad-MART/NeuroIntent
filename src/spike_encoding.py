# Import libraries
import pandas as pd
import numpy as np

# Load normalized features
X = pd.read_csv("data/processed_features.csv")

print("Original Shape:", X.shape)

# Convert values to 0-1 range
X = (X - X.min()) / (X.max() - X.min())

# Rate Coding
# Higher signal value = higher chance of spike
spikes = np.random.rand(*X.shape) < X

# Convert True/False to 1/0
spikes = spikes.astype(int)

print("Spike Shape:", spikes.shape)

# Save spikes
pd.DataFrame(spikes).to_csv(
    "data/spikes.csv",
    index=False
)

print("Spike Encoding Complete")
print("\nSample Spikes:")
print(spikes[:5])
