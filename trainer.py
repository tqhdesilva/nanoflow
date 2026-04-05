"""Training loop, Euler sampler, and checkpointing."""

import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from flow import interpolate, target_velocity


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, losses):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "losses": losses,
    }, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["losses"]


def _make_sigterm_handler(model, optimizer, epoch_ref, losses, path):
    """Returns a SIGTERM handler that saves a checkpoint before exiting.

    Used for graceful preemption on RunPod spot instances.
    Resume with: training.resume=<path>
    """
    def handler(sig, frame):
        print(f"\nSIGTERM at epoch {epoch_ref[0]}, saving to {path}")
        save_checkpoint(path, model, optimizer, epoch_ref[0], losses)
        sys.exit(0)
    return handler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, dataset, cfg, device):
    """
    Train the flow matching model.

    cfg: OmegaConf DictConfig with fields:
        epochs, batch_size, lr, save_every, checkpoint_dir, resume, run_name
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start_epoch = 0
    losses = []

    if cfg.resume:
        start_epoch, losses = load_checkpoint(cfg.resume, model, optimizer)
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # epoch_ref is a mutable cell so the SIGTERM handler always sees current epoch
    epoch_ref = [start_epoch]
    signal.signal(signal.SIGTERM, _make_sigterm_handler(
        model, optimizer, epoch_ref, losses,
        ckpt_dir / f"{cfg.run_name}_preempted.pt",
    ))

    dataset = dataset.to(device)

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training"):
        epoch_ref[0] = epoch
        perm = torch.randperm(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(dataset), cfg.batch_size):
            x_0 = dataset[perm[i:i + cfg.batch_size]]
            eps = torch.randn_like(x_0)
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

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{cfg.run_name}_latest.pt",
                model, optimizer, epoch + 1, losses,
            )

    return losses


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, n_samples=1000, num_steps=100, device="cpu", shape=(2,)):
    """Generate samples via Euler integration of the learned velocity field.

    Starts at t=0 (pure noise), steps to t=1 (data).
    """
    xt = torch.randn(n_samples, *shape, device=device)
    dt = 1.0 / num_steps
    for t in torch.linspace(0, 1, num_steps, device=device):
        t_ = t.expand(n_samples)
        vt = model(xt, t_)
        xt = xt + vt * dt
    return xt
