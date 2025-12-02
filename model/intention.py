
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentionNet(nn.Module):
    def __init__(self, phi_dim, num_latents, hidden_dim=128):
        super(IntentionNet, self).__init__()
        self.fc1 = nn.Linear(phi_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_latents)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)