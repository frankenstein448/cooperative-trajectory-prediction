"""
history_encoder.py
Encoders for the observed-history input: [B, H, 2] past positions (frame-H ..
frame-1, relative to the current position, rotated into the current heading;
see pilot_dataset.PilotDataset.get_history) -> [B, out_dim], the same width as
the CAM embedding. The output is concatenated with the CAM embedding inside
V2XMessageStream (end_to_end_model.py).

  gru:         GRU(2 -> 64), last hidden state -> Linear(64 -> out_dim)
  transformer: Linear(2 -> 64) + learned positional embedding over the H steps,
               2-layer TransformerEncoder (4 heads), mean-pool -> Linear(64 -> out_dim)

Run in gpu-test-venv.
"""
import torch
import torch.nn as nn

HISTORY_ENCODERS = ["gru", "transformer"]
HIDDEN_DIM = 64


class HistoryEncoder(nn.Module):
    def __init__(self, out_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(history)  # h_n: [1, B, hidden]
        return self.out(h_n[-1])     # [B, out_dim]


class TransformerHistoryEncoder(nn.Module):
    def __init__(self, out_dim, history_frames, d_model=HIDDEN_DIM, n_layers=2, n_heads=4):
        super().__init__()
        self.history_frames = history_frames
        self.in_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(history_frames, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        assert history.shape[1] == self.history_frames, (
            f"history has {history.shape[1]} steps, positional embedding has {self.history_frames}")
        x = self.in_proj(history) + self.pos_embed  # [B, H, d_model]
        x = self.encoder(x)
        return self.out(x.mean(dim=1))              # [B, out_dim]


def build_history_encoder(kind, out_dim, history_frames):
    assert kind in HISTORY_ENCODERS, f"unknown history encoder {kind!r}"
    if kind == "gru":
        return HistoryEncoder(out_dim)
    return TransformerHistoryEncoder(out_dim, history_frames)
