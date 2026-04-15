"""Visualization helpers."""

import matplotlib.pyplot as plt


def plot_samples(real, generated, title="Generated vs Real", path="samples.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(real[:, 0], real[:, 1], s=1, alpha=0.5)
    ax1.set_title("Real data")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax2.scatter(generated[:, 0], generated[:, 1], s=1, alpha=0.5)
    ax2.set_title("Generated")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved to {path}")
    plt.close()


def plot_image_samples(samples, title="Generated", path="samples.png"):
    """Plot a grid of generated images. samples: (N, C, H, W) in [-1, 1]."""
    from torchvision.utils import make_grid

    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8, padding=1)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved to {path}")
    plt.close()
