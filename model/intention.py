
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
    def __init__(self, 
                 num_states, 
                 num_actions,
                 num_latents, 
                 emb_dim=16, 
                 rnn_hidden_dim=128, 
                 num_layers=1, 
                 dropout=0.1
    ):
        super(StatesRNN, self).__init__()
        self.state_emb = nn.Embedding(num_states, emb_dim)
        self.action_emb = nn.Embedding(num_actions, emb_dim)
        # self.rnn = nn.RNN(
        #     input_size=emb_dim*3,
        #     hidden_size=rnn_hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )
        self.rnn = nn.LSTM(
            input_size=emb_dim*3,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(rnn_hidden_dim, num_latents)
        
    def forward(self, x):
        # x: (B, seq_len, 3)
        s, a, ns = x.unbind(-1)        # (B, T) * 3
        s_emb = self.state_emb(s)
        a_emb = self.action_emb(a)
        ns_emb = self.state_emb(ns)
        x = torch.cat([s_emb, a_emb, ns_emb], dim=-1)

        rnn_out, _ = self.rnn(x)            # (B, T, rnn_hidden_dim)
        logits = self.fc(rnn_out)           # (B, T, num_latents)

        return logits