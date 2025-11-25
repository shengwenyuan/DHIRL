import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiIntentionR(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(R, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = F.relu

    def forward(self, x, intention_embed):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class R(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(R, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)
    
class Qsh(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(Qsh, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class SAVisitationClassifier(nn.Module):
    """
    A simple feedforward network that takes a state as input
    and outputs logits over all possible discrete actions.
    These logits are used to form π̃_E(a|s) via softmax,
    approximating the expert's state-action visitation distribution.
    """
    def __init__(self, state_dim, n_actions, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], n_actions))  # logits over actions
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        # state: [..., state_dim] → logits: [..., n_actions]
        return self.net(state)