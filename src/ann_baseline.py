import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================
# Load Data
# ==========================

X = pd.read_csv("../data/processed_features.csv").values

y = pd.read_csv(
    "../data/processed_labels.csv"
).values.ravel()

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Convert To Tensors
# ==========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

# ==========================
# ANN Model
# ==========================

model = nn.Sequential(

    nn.Linear(64, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 4)
)

# ==========================
# Loss + Optimizer
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================
# Training
# ==========================

epochs = 30

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {loss.item():.4f}"
    )

# ==========================
# Evaluation
# ==========================

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

print("\nANN Accuracy:", acc)
