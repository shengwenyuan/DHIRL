
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


GATE_MODES = ('retrospective', 'causal', 'state_only')


def _validate_gate_mode(gate_mode):
    if gate_mode not in GATE_MODES:
        raise ValueError(f'Unknown gate_mode={gate_mode!r}; expected one of {GATE_MODES}')


def _combine_gate_inputs(state_embeds, action_embeds, gate_mode):
    if gate_mode == 'retrospective':
        return state_embeds + action_embeds
    if gate_mode == 'state_only':
        return state_embeds

    previous_action_embeds = torch.zeros_like(action_embeds)
    previous_action_embeds[:, 1:, :] = action_embeds[:, :-1, :]
    return state_embeds + previous_action_embeds


def _causal_attention_mask(sequence_length, device):
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


class IntentionRNN(nn.Module):
    def __init__(self, num_states, num_actions, num_latents, hidden_dim=128, rnn_hidden_dim=128, num_layers=1, dropout=0.1, gate_mode='retrospective'):
        super(IntentionRNN, self).__init__()
        _validate_gate_mode(gate_mode)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.gate_mode = gate_mode
        
        self.state_embed = nn.Embedding(num_states, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Linear(rnn_hidden_dim, num_latents)

    def forward(self, bs, ba, mask=None, total_length=None):
        state_embeds = self.state_embed(bs)   # (B, T, hidden_dim)
        action_embeds = self.action_embed(ba) # (B, T, hidden_dim)
        x = _combine_gate_inputs(state_embeds, action_embeds, self.gate_mode)
        if mask is not None:
            lengths = mask.sum(dim=1)
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_out_packed, _ = self.rnn(x_packed)
            rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True, total_length=total_length)  # (B, T_max, rnn_hidden_dim)
        else:
            rnn_out, _ = self.rnn(x)
        logits = self.output_proj(rnn_out)           # (B, T, num_latents)

        return logits

    def candidate_action_logits(self, bs, ba, mask=None):
        """Evaluate every current action while advancing only the observed path."""
        if self.gate_mode != 'retrospective':
            raise ValueError(
                'candidate_action_logits is defined only for the retrospective gate.'
            )
        if self.training:
            raise RuntimeError('candidate_action_logits requires eval mode.')
        if bs.ndim != 2 or ba.shape != bs.shape:
            raise ValueError('bs and ba must have matching (batch, time) shapes.')
        if bs.numel() == 0:
            raise ValueError('candidate_action_logits requires non-empty inputs.')
        if mask is not None and mask.shape != bs.shape:
            raise ValueError('mask must match the (batch, time) input shape.')
        if torch.any(bs < 0) or torch.any(bs >= self.state_embed.num_embeddings):
            raise ValueError('State index is outside the embedding support.')
        if torch.any(ba < 0) or torch.any(ba >= self.action_embed.num_embeddings):
            raise ValueError('Action index is outside the embedding support.')
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError('mask must have boolean dtype.')
            lengths = mask.sum(dim=1)
            if torch.any(lengths <= 0):
                raise ValueError('Every sequence needs at least one valid step.')
            prefix_mask = (
                torch.arange(bs.shape[1], device=bs.device)[None, :]
                < lengths[:, None]
            )
            if not torch.equal(mask, prefix_mask):
                raise ValueError('mask rows must be contiguous valid prefixes.')

        batch_size, sequence_length = bs.shape
        num_actions = self.action_embed.num_embeddings
        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.rnn_hidden_dim,
            dtype=self.state_embed.weight.dtype,
            device=bs.device,
        )
        all_action_embeds = self.action_embed.weight
        batch_indices = torch.arange(batch_size, device=bs.device)
        candidate_logits = []

        for step in range(sequence_length):
            state_embed = self.state_embed(bs[:, step])
            candidate_inputs = (
                state_embed[:, None, :] + all_action_embeds[None, :, :]
            )
            candidate_hidden = (
                hidden[:, :, None, :]
                .expand(-1, -1, num_actions, -1)
                .reshape(self.num_layers, batch_size * num_actions, self.rnn_hidden_dim)
                .contiguous()
            )
            candidate_output, next_candidate_hidden = self.rnn(
                candidate_inputs.reshape(batch_size * num_actions, 1, -1),
                candidate_hidden,
            )
            step_logits = self.output_proj(candidate_output[:, 0, :]).reshape(
                batch_size, num_actions, -1
            )
            candidate_logits.append(step_logits)

            observed_indices = batch_indices * num_actions + ba[:, step]
            observed_hidden = next_candidate_hidden[:, observed_indices, :]
            if mask is None:
                hidden = observed_hidden
            else:
                valid = mask[:, step].reshape(1, batch_size, 1)
                hidden = torch.where(valid, observed_hidden, hidden)

        return torch.stack(candidate_logits, dim=1)


class IntentionLSTM(nn.Module):
    def __init__(self, num_states, num_actions, num_latents, hidden_dim=128, rnn_hidden_dim=128, num_layers=1, dropout=0.1, gate_mode='retrospective'):
        super(IntentionLSTM, self).__init__()
        _validate_gate_mode(gate_mode)
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.gate_mode = gate_mode

        self.state_embed = nn.Embedding(num_states, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, hidden_dim)

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_proj = nn.Linear(rnn_hidden_dim, num_latents)

    def forward(self, bs, ba, mask=None, total_length=None):
        state_embeds = self.state_embed(bs)   # (B, T, hidden_dim)
        action_embeds = self.action_embed(ba) # (B, T, hidden_dim)
        x = _combine_gate_inputs(state_embeds, action_embeds, self.gate_mode)
        if mask is not None:
            lengths = mask.sum(dim=1)
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_out_packed, _ = self.rnn(x_packed)
            rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True, total_length=total_length)  # (B, T_max, rnn_hidden_dim)
        else:
            rnn_out, _ = self.rnn(x)
        logits = self.output_proj(rnn_out)           # (B, T, num_latents)

        return logits


class IntentionTransformer(nn.Module):
    def __init__(self,
                 num_states,
                 num_actions,
                 num_latents,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1,
                 gate_mode='retrospective'):
        super().__init__()
        _validate_gate_mode(gate_mode)
        self.gate_mode = gate_mode
        self.state_embed = nn.Embedding(num_states, d_model)
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_latents)

    def forward(self, bs, ba, mask=None, total_length=None):
        # bs: (B, T), ba: (B, T)
        state_embeds = self.state_embed(bs)   # (B, T, d_model)
        action_embeds = self.action_embed(ba) # (B, T, d_model)
        x = _combine_gate_inputs(state_embeds, action_embeds, self.gate_mode)
        x = self.pos_encoding(x)
        attention_mask = None
        if self.gate_mode != 'retrospective':
            attention_mask = _causal_attention_mask(x.size(1), x.device)
        if mask is not None:
            padding_mask = ~mask
            x = self.transformer(x, mask=attention_mask, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x, mask=attention_mask)  # (B, T, d_model)

        logits = self.fc_out(x)           # (B, T, num_latents)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
