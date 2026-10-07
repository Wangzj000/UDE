import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, args, in_channels=1, hidden=40, layers=2, out_channels=1):
        super().__init__()
        self.rnn = nn.GRU(in_channels, hidden, layers, batch_first=True)
        self.mu  = nn.Linear(hidden, out_channels)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: (B, T, C)
        out,_ = self.rnn(x)
        return self.mu(out)        