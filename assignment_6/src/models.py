from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SeqClassifier(nn.Module):
    def __init__(
        self,
        rnn_type: Literal["rnn", "lstm"] = "rnn",
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        num_classes: int = 5,
        model_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        if model_type is not None:
            rnn_type = model_type
        if rnn_type not in ("rnn", "lstm"):
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'; expected 'rnn' or 'lstm'.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.rnn_type = rnn_type
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        rnn_dropout = float(dropout) if self.num_layers > 1 else 0.0
        kwargs = dict(
            input_size=int(input_size),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
        )
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(nonlinearity="tanh", **kwargs)
        else:
            self.rnn = nn.LSTM(**kwargs)

        self.out_dropout = nn.Dropout(float(dropout)) if (self.num_layers == 1 and float(dropout) > 0.0) else nn.Identity()
        self.classifier = nn.Linear(self.hidden_size * self.num_directions, int(num_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.detach().to(device="cpu", dtype=torch.int64), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)

        h_n = hidden[0] if self.rnn_type == "lstm" else hidden
        last_layer = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)[-1]
        if self.bidirectional:
            feat = torch.cat([last_layer[0], last_layer[1]], dim=-1)
        else:
            feat = last_layer[0]

        return self.classifier(self.out_dropout(feat))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
