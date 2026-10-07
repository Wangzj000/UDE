import torch, torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # 记下来，后面要裁
        self.conv = nn.Conv1d(
            channels, 2*channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation
        )
        self.gate = nn.GLU(dim=1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        y = self.conv(x)
        if self.pad:                       # 只截掉右边
            y = y[..., :-self.pad]         # (B, 2C, T)
        y = self.gate(y)                   # (B, C, T)
        return self.skip(y) + x, y         # 残差 & 跳连

class Model(nn.Module):
    def __init__(self, args, n_targets=1, horizon=96,
                 in_channels=1, res_channels=32,
                 kernel_size=2, dilations=(1,2,4,8,16,32)):
        super().__init__()
        self.horizon  = horizon
        self.n_targets = n_targets

        self.causal_in = nn.Conv1d(in_channels, res_channels, 1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(res_channels, kernel_size, d)
             for d in dilations]
        )
        # --------- 关键：把输出通道设成 n_targets ----------
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(res_channels, res_channels, 1),
            nn.ReLU(),
            nn.Conv1d(res_channels, n_targets, 1)     # <‑‑ 这里
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x : (B, seq_len, in_channels)
        return : (B, horizon, n_targets)  == (B,T,C)
        """
        x = x.transpose(1,2)               # (B,C,T)

        skip = 0
        x = self.causal_in(x)
        for blk in self.blocks:
            x, s = blk(x)
            skip = skip + s

        y = self.proj(skip)                # (B, n_targets, T)
        # 取最后 horizon 步 → (B, n_targets, horizon)
        y = y[..., -self.horizon:]
        # 转回 (B, horizon, n_targets) == (B,T,C)
        return y.transpose(1,2)

