# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Store all datasets
dfs = []

# Load all CSV files
for i in range(4):
    df = pd.read_csv(f"data/{i}.csv", header=None)
    dfs.append(df)

# Merge datasets into one dataframe
data = pd.concat(dfs, ignore_index=True)

print("Combined Shape:", data.shape)

# Features (first 64 columns)
X = data.iloc[:, :-1]

# Labels (last column)
y = data.iloc[:, -1]

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)

# Normalize EMG signals
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nNormalization Complete")

# Save processed data
pd.DataFrame(X_scaled).to_csv(
    "data/processed_features.csv",
    index=False
)

pd.DataFrame(y).to_csv(
    "data/processed_labels.csv",
    index=False
)

print("Processed files saved.")
