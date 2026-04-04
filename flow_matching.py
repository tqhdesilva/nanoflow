"""
NanoFlow — minimal flow matching in pure PyTorch.
Single file: model, training, sampling, visualization.
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from tqdm import tqdm


# ---------------------------------------------------------------------------
# A: Dataset
# ---------------------------------------------------------------------------

def moons_dataset(n=8000, noise=0.05):
    """2D moons dataset, normalized to roughly [-1, 1]."""
    X, _ = make_moons(n_samples=n, noise=noise)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize
    return torch.tensor(X, dtype=torch.float32)


def fashion_dataset():
    """FashionMNIST, scaled to [-1, 1]. Returns (60000, 1, 28, 28) tensor."""
    from torchvision import datasets, transforms
    ds = datasets.FashionMNIST(
        root="./data", train=True, download=True,
        transform=transforms.ToTensor(),
    )
    X = torch.stack([img for img, _ in ds])  # (60000, 1, 28, 28) in [0, 1]
    return X * 2 - 1  # scale to [-1, 1]


# ---------------------------------------------------------------------------
# B: Sinusoidal time embedding
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
# C: MLP model (adapted from tiny-diffusion)
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
            h = block(h) + h  # residual
        return self.output_proj(h)


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
    Tiny UNet for 28×28 grayscale: 2 down/up levels, no attention.
    Takes (x, t) and outputs predicted velocity (same shape as x).
    x: (B, 1, 28, 28)   t: (B,)   out: (B, 1, 28, 28)
    """
    def __init__(self, base_ch=32, time_dim=64):
        super().__init__()
        # Time embedding: sinusoidal → 2-layer MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )
        # Encoder
        self.conv_in = nn.Conv2d(1, base_ch, 3, padding=1)
        self.enc0 = ResBlock(base_ch, base_ch, time_dim)
        self.down0 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)       # 28→14
        self.enc1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down1 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)  # 14→7
        # Bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)   # 7→14
        self.dec1 = ResBlock(base_ch * 4, base_ch, time_dim)  # cat with skip
        self.up0 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)           # 14→28
        self.dec0 = ResBlock(base_ch * 2, base_ch, time_dim)  # cat with skip
        self.conv_out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h0 = self.enc0(self.conv_in(x), t_emb)
        h1 = self.enc1(self.down0(h0), t_emb)
        h = self.mid(self.down1(h1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb)
        h = self.dec0(torch.cat([self.up0(h), h0], dim=1), t_emb)
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# D: Flow matching core — YOUR CODE HERE
# ---------------------------------------------------------------------------

def interpolate(x_0, eps, t):
    """
    Compute x_t along the flow matching interpolation path.

    x_0: (B, 2) — data samples
    eps: (B, 2) — noise samples ~ N(0, I)
    t:   (B, 1) — time in [0, 1], already unsqueezed

    Returns: x_t (B, 2)

    At t=0: x_t = eps (pure noise)
    At t=1: x_t = x_0 (clean data)
    """
    # CondOT path
    return (1 - t) * eps + t * x_0


def target_velocity(x_0, eps):
    """
    Compute the ground-truth velocity for flow matching.

    x_0: (B, 2) — data samples
    eps: (B, 2) — noise samples

    Returns: v (B, 2) — the velocity that moves eps toward x_0
    """
    return x_0 - eps


# ---------------------------------------------------------------------------
# E: Training — YOUR CODE HERE
# ---------------------------------------------------------------------------

def train(model, dataset, num_epochs=300, batch_size=256, lr=1e-3, device="cpu"):
    """
    Train the flow matching model.

    Core loop (each step):
    1. Sample a batch of data x_0
    2. Sample noise eps ~ N(0, I)
    3. Sample t ~ Uniform(0, 1)
    4. Compute x_t = interpolate(x_0, eps, t)
    5. Predict velocity: v_pred = model(x_t, t)
    6. Loss = MSE(v_pred, target_velocity(x_0, eps))
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = dataset.to(device)
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Shuffle and batch
        perm = torch.randperm(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(dataset), batch_size):
            x_0 = dataset[perm[i:i + batch_size]]

            eps = torch.randn_like(x_0, device=device)
            t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=device)
            xt = interpolate(x_0, eps, t)
            v_pred = model(xt, t.view(-1))
            vt = target_velocity(x_0, eps)
            loss = nn.functional.mse_loss(v_pred, vt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses


# ---------------------------------------------------------------------------
# F: Euler ODE sampler — YOUR CODE HERE
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, n_samples=1000, num_steps=100, device="cpu", shape=(2,)):
    """
    Generate samples via Euler integration of the learned velocity field.

    Start at t=0 (pure noise), step to t=1 (data).
    dt = 1/num_steps

    Loop:
        t = step / num_steps
        v = model(x, t)
        x = x + v * dt

    Returns: (n_samples, *shape) tensor of generated samples.
    """
    xt = torch.randn(n_samples, *shape, device=device)  # N(0, I)
    dt = 1.0 / num_steps
    for t in torch.linspace(0, 1, num_steps, device=device):
        t_ = t.expand(n_samples)
        vt = model(xt, t_)
        xt = xt + vt * dt
    return xt


# ---------------------------------------------------------------------------
# G: Visualization + main
# ---------------------------------------------------------------------------

def plot_samples(real, generated, title="Generated vs Real", path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(real[:, 0], real[:, 1], s=1, alpha=0.5)
    ax1.set_title("Real data")
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
    ax2.scatter(generated[:, 0], generated[:, 1], s=1, alpha=0.5)
    ax2.set_title("Generated")
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
    plt.suptitle(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        print(f"Saved to {path}")
    else:
        plt.show()
    plt.close()


def plot_image_samples(samples, title="Generated", path=None):
    """Plot a grid of generated images. samples: (N, 1, 28, 28) in [-1, 1]."""
    from torchvision.utils import make_grid
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8, padding=1)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
    else:
        plt.show()
    plt.close()


def plot_loss(losses, path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss")
    if path:
        plt.savefig(path, dpi=150)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoFlow — minimal flow matching")
    parser.add_argument("--dataset", choices=["moons", "fashion"], default="moons")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=100, help="Euler integration steps")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing")
    args = parser.parse_args()

    # Dataset-specific defaults
    if args.dataset == "moons":
        defaults = dict(epochs=300, batch_size=256, n_samples=1000)
    else:
        defaults = dict(epochs=20, batch_size=128, n_samples=64)
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    # Data + model
    if args.dataset == "moons":
        dataset = moons_dataset()
        model = MLP().to(args.device)
        sample_shape = (2,)
    else:
        dataset = fashion_dataset()
        model = UNet().to(args.device)
        sample_shape = (1, 28, 28)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    losses = train(model, dataset, args.epochs, args.batch_size, args.lr, args.device)

    # Sample
    generated = sample(model, args.n_samples, args.num_steps, args.device, shape=sample_shape)

    # Visualize
    plot_loss(losses, "loss.png" if args.save else None)
    if args.dataset == "moons":
        plot_samples(dataset[:1000].cpu(), generated.cpu(),
                     f"After {args.epochs} epochs, {args.num_steps} Euler steps",
                     "samples.png" if args.save else None)
    else:
        plot_image_samples(generated,
                           f"After {args.epochs} epochs, {args.num_steps} Euler steps",
                           "samples.png" if args.save else None)
