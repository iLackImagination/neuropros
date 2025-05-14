import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 12)
        )

    def forward(self, x):
        #x = self.flatten(x)
        y = self.linear_stack(x)
        return y