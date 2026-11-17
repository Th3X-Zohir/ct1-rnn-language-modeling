"""Character-level recurrent language models.

``CharRNN`` = Embedding -> {RNN | GRU | LSTM} (stacked) -> Dropout -> Linear(vocab).
The same class is used for all three cell types so that the comparison changes
only the recurrent cell, nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, cell: str = "lstm", emb_dim: int = 64,
                 hidden: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab_size, emb_dim)
        kw = dict(nonlinearity="tanh") if cell == "rnn" else {}
        self.rnn = CELLS[cell](emb_dim, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0, **kw)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, state=None):
        """x: (B, T) int64 -> logits (B, T, V), new recurrent state."""
        h, state = self.rnn(self.embed(x), state)
        return self.head(self.drop(h)), state


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
