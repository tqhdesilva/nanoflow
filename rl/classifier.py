"""Small CNN classifier for FashionMNIST.

Used as the reward model for Flow-GRPO. Train with:
    uv run python -m rl.classifier --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FashionMNISTDataset


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


def load_classifier(checkpoint: str, device: torch.device) -> FashionCNN:
    model = FashionCNN().to(device)
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train(epochs: int = 5, batch_size: int = 128, lr: float = 1e-3,
          out_path: str = "runs/reward_models/fashion_classifier.pt",
          device: str = "mps") -> None:
    dev = torch.device(device)
    train_ds = FashionMNISTDataset(train=True)
    val_ds = FashionMNISTDataset(train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    model = FashionCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"epoch {ep+1}/{epochs} val_acc={correct/total:.4f}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved classifier to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/reward_models/fashion_classifier.pt")
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.out, args.device)


if __name__ == "__main__":
    main()
