
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
    

class StatesRNN(nn.Module):
    def __init__(self, phi_dim, num_latents, hidden_dim=128, rnn_hidden_dim=128, num_layers=1):
        super(StatesRNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(phi_dim, hidden_dim)
        
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        self.output_proj = nn.Linear(rnn_hidden_dim, num_latents)

    def forward(self, x):
        # x: (batch_size, seq_len, phi_dim)
        x = F.relu(self.input_proj(x))               # (B, T, hidden_dim)
        rnn_out, _ = self.rnn(x)                     # (B, T, rnn_hidden_dim)
        logits = self.output_proj(rnn_out)           # (B, T, num_latents)

        return logits