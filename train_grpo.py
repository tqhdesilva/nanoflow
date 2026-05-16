"""Flow-GRPO training entry point.

Outer rollout / inner-update loop. Talks only to a `RolloutClient`; never
imports the SDE sampler directly.
"""

from __future__ import annotations

import copy
import signal
import sys

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_

import config as _config  # noqa: F401 — registers structured config schema
from callbacks import CheckpointCallback, RunDirCallback
from rl.grpo import compute_group_advantage, grpo_loss
from rl.rollout_client import RolloutClient
from rl.sde_sampler import recompute_logprobs


def load_seed_policy(model_cfg, checkpoint: str, device: torch.device) -> nn.Module:
    model: nn.Module = hydra.utils.instantiate(model_cfg).to(device)
    ckpt = torch.load(checkpoint, weights_only=True, map_location=device)
    state = ckpt.get("ema_state") or ckpt["model_state"]
    model.load_state_dict(state)
    tag = "EMA" if ckpt.get("ema_state") else "model"
    epoch = ckpt.get("train_progress", {}).get("num_epochs_completed", "?")
    print(f"Loaded {tag} seed weights from {checkpoint} (epoch {epoch})")
    return model


def sample_prompts(
    batch_size: int, G: int, num_classes: int, device: torch.device
) -> torch.Tensor:
    base = torch.randint(0, num_classes, (batch_size,), device=device)
    return base.repeat_interleave(G)


@hydra.main(config_path="configs", config_name="config_grpo", version_base=None)
def main(cfg) -> None:
    device = torch.device(cfg.device)
    rl = cfg.rl_training

    policy = load_seed_policy(cfg.model, cfg.seed_checkpoint, device)
    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    reward_fn = hydra.utils.instantiate(cfg.reward)
    rollout_client: RolloutClient = hydra.utils.instantiate(
        cfg.rollout_client,
        model=policy,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=rl.lr)

    run_dir_cb = RunDirCallback(
        runs_dir=cfg.runs_dir,
        run_prefix=rl.run_prefix,
        cfg=cfg,
    )
    ckpt_cb = CheckpointCallback(
        ckpt_dir=run_dir_cb.ckpt_dir,
        save_every=rl.save_every,
        resume=None,
    )
    writer = run_dir_cb.writer

    def _save(name: str) -> None:
        torch.save(
            {
                "model_state": policy.state_dict(),
                "optimizer_state": opt.state_dict(),
                "train_progress": {"num_epochs_completed": epoch},
            },
            ckpt_cb.save_path(name),
        )

    epoch = 0

    def _sigterm(_sig, _frame):
        print(f"\nSIGTERM caught — saving preempted checkpoint")
        _save("preempted")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    global_step = 0
    for epoch in range(rl.epochs):
        # Rollout phase
        prompts = sample_prompts(rl.batch_size, rl.G, rl.num_classes, device)
        traj = rollout_client.rollout(prompts)
        rewards = reward_fn(traj.xs[-1], traj.prompts)
        advantage = compute_group_advantage(rewards, rl.G)

        # Inner update (PPO-style)
        loss, info = None, None
        for inner in range(rl.num_inner):
            new_logprobs, new_mus = recompute_logprobs(
                policy, traj, rl.sampler, no_grad=False
            )
            _, ref_mus = recompute_logprobs(ref_policy, traj, rl.sampler, no_grad=True)
            loss, info = grpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=traj.log_probs_old,
                new_mus=new_mus,
                ref_mus=ref_mus,
                sigma_step=traj.stds,
                advantage=advantage,
                clip_eps=rl.clip_eps,
                kl_beta=rl.kl_beta,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if rl.grad_clip > 0:
                clip_grad_norm_(policy.parameters(), rl.grad_clip)
            opt.step()

            if writer is not None:
                writer.add_scalar("grpo/loss", info["loss"].item(), global_step)
                writer.add_scalar("grpo/pg_loss", info["pg_loss"].item(), global_step)
                writer.add_scalar("grpo/kl", info["kl"].item(), global_step)
                writer.add_scalar(
                    "grpo/approx_kl_is", info["approx_kl_is"].item(), global_step
                )
                writer.add_scalar(
                    "grpo/clip_frac", info["clip_frac"].item(), global_step
                )
                writer.add_scalar(
                    "grpo/ratio_mean", info["ratio_mean"].item(), global_step
                )
            global_step += 1

        rollout_client.update_weights(policy.state_dict())

        if writer is not None and epoch % rl.log_every == 0:
            writer.add_scalar("rl/reward_mean", rewards.mean().item(), epoch)
            writer.add_scalar("rl/reward_std", rewards.std().item(), epoch)
            writer.add_scalar(
                "rl/advantage_abs_mean", advantage.abs().mean().item(), epoch
            )
            if info:
                print(
                    f"epoch {epoch}/{rl.epochs} | reward={rewards.mean().item():.4f} "
                    f"| kl={info['kl'].item():.4f} | pg={info['pg_loss'].item():.4f}"
                )

        if (epoch + 1) % rl.save_every == 0:
            _save("latest")

    _save("latest")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
