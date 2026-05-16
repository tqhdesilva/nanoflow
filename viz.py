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


def plot_image_samples(
    samples,
    title="Generated",
    path="samples.png",
    labels=None,
    class_names=None,
):
    """Plot a grid of generated images. samples: (N, C, H, W) in [-1, 1]."""
    import math

    samples = samples.clamp(-1, 1) * 0.5 + 0.5
    n_samples = samples.shape[0]
    label_values = labels.cpu().tolist() if labels is not None else None
    if label_values is not None and label_values[:10] == list(range(10)):
        nrow = min(10, n_samples)
    else:
        nrow = min(8, n_samples)
    ncol = math.ceil(n_samples / nrow)

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol * 1.15))
    if ncol == 1:
        axes = [axes]

    for i in range(ncol * nrow):
        ax = axes[i // nrow][i % nrow]
        ax.axis("off")
        if i >= n_samples:
            continue

        image = samples[i].cpu()
        if image.shape[0] == 1:
            ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(image.permute(1, 2, 0).numpy())
        if label_values is not None:
            label = label_values[i]
            if class_names is not None and 0 <= label < len(class_names):
                label = class_names[label]
            ax.set_title(str(label), fontsize=7, pad=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved to {path}")
    plt.close(fig)
