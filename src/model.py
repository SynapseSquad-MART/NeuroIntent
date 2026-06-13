import torch
import torch.nn as nn
import norse.torch as norse


class SNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(64, 128)

        self.lif = norse.LIFCell()

        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):

        x = self.fc1(x)

        x, _ = self.lif(x)

        x = self.fc2(x)

        return x
