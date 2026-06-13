import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import SNN

# Load spike data
X = pd.read_csv("../data/spikes.csv").values

# Load labels
y = pd.read_csv("../data/processed_labels.csv").values.ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create model
model = SNN()

# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {loss.item():.4f}"
    )

# Evaluation
with torch.no_grad():

    predictions = model(X_test)

    predicted = torch.argmax(
        predictions,
        dim=1
    )

acc = accuracy_score(
    y_test.numpy(),
    predicted.numpy()
)

print("\nAccuracy:", acc)

torch.save(
    model.state_dict(),
    "../models/snn_model.pth"
)

print("Model Saved")
