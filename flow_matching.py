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
    raise NotImplementedError("TODO: implement linear interpolation (1-t)*eps + t*x_0")


def target_velocity(x_0, eps):
    """
    Compute the ground-truth velocity for flow matching.

    x_0: (B, 2) — data samples
    eps: (B, 2) — noise samples

    Returns: v (B, 2) — the velocity that moves eps toward x_0
    """
    raise NotImplementedError("TODO: implement target velocity x_0 - eps")


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

            # TODO: implement the training step
            # 1. Sample eps ~ N(0, I) matching x_0 shape
            # 2. Sample t ~ U(0, 1), shape (B, 1) for broadcasting
            # 3. Compute x_t via interpolate()
            # 4. Forward pass: v_pred = model(x_t, t.squeeze(-1))
            #    (squeeze t back to (B,) for the model's time embedding)
            # 5. Compute MSE loss against target_velocity()
            # 6. optimizer.zero_grad(), loss.backward(), optimizer.step()
            raise NotImplementedError("TODO: implement training step")

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses


# ---------------------------------------------------------------------------
# F: Euler ODE sampler — YOUR CODE HERE
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, n_samples=1000, num_steps=100, device="cpu"):
    """
    Generate samples via Euler integration of the learned velocity field.

    Start at t=0 (pure noise), step to t=1 (data).
    dt = 1/num_steps

    Loop:
        t = step / num_steps
        v = model(x, t)
        x = x + v * dt

    Returns: (n_samples, 2) tensor of generated points.
    """
    raise NotImplementedError("TODO: implement Euler ODE sampler")


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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=100, help="Euler integration steps")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing")
    args = parser.parse_args()

    # Data
    dataset = moons_dataset()

    # Model
    model = MLP().to(args.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    losses = train(model, dataset, args.epochs, args.batch_size, args.lr, args.device)

    # Sample
    generated = sample(model, args.n_samples, args.num_steps, args.device)

    # Visualize
    plot_loss(losses, "loss.png" if args.save else None)
    plot_samples(dataset[:1000].cpu(), generated.cpu(),
                 f"After {args.epochs} epochs, {args.num_steps} Euler steps",
                 "samples.png" if args.save else None)
