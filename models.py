"""Model architectures — MLP (2D toy), UNet (images)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (shared by MLP and UNet)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,) float in [0, 1] → (B, dim) embedding."""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Scale t to [0, 1000] for better frequency coverage
        emb = (t[:, None] * 1000.0) * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# MLP (for 2D toy data)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class MLP(nn.Module):
    """
    Takes (x, t) and outputs predicted velocity.
    x: (B, 2) — 2D point
    t: (B,)   — time in [0, 1]
    out: (B, 2) — predicted velocity vector
    """
    def __init__(self, hidden_dim=128, num_layers=4, time_dim=32):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.input_proj = nn.Linear(2 + time_dim, hidden_dim)
        self.blocks = nn.ModuleList([Block(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):
        t_emb = self.time_embed(t)           # (B, time_dim)
        h = torch.cat([x, t_emb], dim=-1)    # (B, 2 + time_dim)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h) + h                 # residual
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# UNet (for images, generalized depth)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Conv residual block with time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.gelu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Generalized UNet for image flow matching.

    Args:
        in_ch:    input/output image channels (1=grayscale, 3=RGB)
        base_ch:  channels at level 0; doubles each level
        depth:    number of downsampling levels
        time_dim: time embedding dimension
        use_attn: self-attention at bottleneck (recommended for >= 32x32 images)

    Channel progression: [base_ch, base_ch*2, ..., base_ch*(2**(depth-1))]
    Spatial: H → H/2 → ... → H/(2**depth) at bottleneck
    """
    def __init__(self, in_ch=1, base_ch=32, depth=2, time_dim=64, use_attn=False):
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(depth)]

        # Time embedding: sinusoidal → 2-layer MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            ResBlock(channels[i], channels[i], time_dim) for i in range(depth)
        ])
        # down[i]: channels[i] → channels[i+1] (or channels[i] for last level → bottleneck)
        self.downs = nn.ModuleList([
            nn.Conv2d(
                channels[i],
                channels[i + 1] if i < depth - 1 else channels[i],
                3, stride=2, padding=1,
            )
            for i in range(depth)
        ])

        # Bottleneck
        bot_ch = channels[-1]
        self.mid = ResBlock(bot_ch, bot_ch, time_dim)

        # Optional self-attention at bottleneck (pre-norm + residual)
        self.use_attn = use_attn
        if use_attn:
            self.attn = nn.MultiheadAttention(bot_ch, num_heads=8, batch_first=True)
            self.attn_norm = nn.LayerNorm(bot_ch)

        # Decoder (reversed encoder levels)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i], 2, stride=2)
            for i in range(depth - 1, -1, -1)
        ])
        # dec[j] at level i (j = depth-1-i): cat skip → channels[i]*2 → channels[i-1]
        self.dec_blocks = nn.ModuleList([
            ResBlock(channels[i] * 2, channels[i - 1] if i > 0 else channels[0], time_dim)
            for i in range(depth - 1, -1, -1)
        ])

        self.conv_out = nn.Conv2d(channels[0], in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoder: collect skip connections
        h = self.conv_in(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid(h, t_emb)
        if self.use_attn:
            B, C, H, W = h.shape
            h_flat = h.view(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
            h_normed = self.attn_norm(h_flat)
            h_attn, _ = self.attn(h_normed, h_normed, h_normed)
            h = (h_flat + h_attn).permute(0, 2, 1).view(B, C, H, W)

        # Decoder: upsample + cat skip + resblock
        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1), t_emb)

        return self.conv_out(h)
