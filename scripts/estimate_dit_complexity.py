#!/usr/bin/env python3
"""Estimate DiT params, active params, and forward MACs from Hydra configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

import config as _config  # noqa: F401, registers config schemas
from models_dit import (
    ClassCondDeferredMaskingDiT,
    ClassCondDiT,
    DenseFFN,
    DiTBlock,
    ExpertChoiceMoEFFN,
    PackedExpertChoiceMoEFFN,
)


@dataclass(frozen=True)
class Complexity:
    name: str
    experiment: str
    total_params: int
    active_params: float
    train_macs: float
    eval_macs: float

    @property
    def train_flops(self) -> float:
        return 6.0 * self.train_macs

    @property
    def forward_flops(self) -> float:
        return 2.0 * self.train_macs

    def row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "experiment": self.experiment,
            "total_params_m": self.total_params / 1e6,
            "active_params_m": self.active_params / 1e6,
            "train_macs_g": self.train_macs / 1e9,
            "eval_macs_g": self.eval_macs / 1e9,
            "train_flops_g": self.train_flops / 1e9,
        }


def compose_model_config(experiment: str) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    return cfg.model


def mutate_moe_config(
    model_cfg: DictConfig,
    *,
    name: str,
    moe_width_scale: float | None = None,
    ffn_width_scale: float | None = None,
    expert_capacity: float | None = None,
    dense_peer: bool = False,
    patch_mixer_dense: bool = False,
    backbone_depth: int | None = None,
    patch_mixer_depth: int | None = None,
) -> tuple[str, DictConfig]:
    cfg = copy.deepcopy(model_cfg)
    OmegaConf.set_struct(cfg, False)

    def visit(node: Any) -> None:
        if isinstance(node, ListConfig):
            for item in node:
                visit(item)
            return
        if not isinstance(node, DictConfig):
            return
        target = node.get("_target_")
        in_patch_mixer = any(part == "patch_mixer" for part in node._get_full_key(None).split("."))
        if target in {"models_dit.DenseFFN", "models_dit.ExpertChoiceMoEFFN"}:
            if ffn_width_scale is not None:
                node.mlp_width = max(1, int(round(int(node.mlp_width) * ffn_width_scale)))
        if target == "models_dit.ExpertChoiceMoEFFN":
            if dense_peer or (patch_mixer_dense and in_patch_mixer):
                node._target_ = "models_dit.DenseFFN"
                for key in ["num_experts", "expert_capacity", "collect_routing_stats"]:
                    if key in node:
                        del node[key]
            else:
                if moe_width_scale is not None:
                    node.mlp_width = max(1, int(round(int(node.mlp_width) * moe_width_scale)))
                if expert_capacity is not None:
                    node.expert_capacity = float(expert_capacity)
        for value in node.values():
            visit(value)

    visit(cfg)
    if backbone_depth is not None:
        cfg.backbone.blocks = cfg.backbone.blocks[:backbone_depth]
    if patch_mixer_depth is not None and cfg.get("patch_mixer") is not None:
        if patch_mixer_depth <= 0:
            cfg.patch_mixer = None
        else:
            cfg.patch_mixer.blocks = cfg.patch_mixer.blocks[:patch_mixer_depth]
    return name, cfg


def instantiate_meta(model_cfg: DictConfig):
    with torch.device("meta"):
        return instantiate(model_cfg)


def estimate(name: str, experiment: str, model_cfg: DictConfig) -> Complexity:
    model = instantiate_meta(model_cfg)
    total_params = sum(param.numel() for param in model.parameters())
    active_params = estimate_active_params(model)
    train_macs, eval_macs = estimate_macs(model)
    return Complexity(
        name=name,
        experiment=experiment,
        total_params=total_params,
        active_params=active_params,
        train_macs=train_macs,
        eval_macs=eval_macs,
    )


def estimate_active_params(model: torch.nn.Module) -> float:
    total = float(sum(param.numel() for param in model.parameters()))
    active = total
    for _, module in model.named_modules():
        if isinstance(module, ExpertChoiceMoEFFN):
            expert_params = sum(param.numel() for param in module.experts[0].parameters())
            all_experts = sum(param.numel() for param in module.experts.parameters())
            router = sum(param.numel() for param in module.router.parameters())
            token_multiplier = active_expert_multiplier(module, num_tokens=64)
            active_moe = router + token_multiplier * expert_params
            total_moe = router + all_experts
            active += active_moe - total_moe
        elif isinstance(module, PackedExpertChoiceMoEFFN):
            expert_params = packed_expert_param_count(module)
            all_experts = module.num_experts * expert_params
            router = sum(param.numel() for param in module.router.parameters())
            token_multiplier = active_expert_multiplier(module, num_tokens=64)
            active_moe = router + token_multiplier * expert_params
            total_moe = router + all_experts
            active += active_moe - total_moe
    return active


def active_expert_multiplier(
    module: ExpertChoiceMoEFFN | PackedExpertChoiceMoEFFN,
    num_tokens: int,
) -> float:
    k = module._tokens_per_expert(num_tokens)
    return module.num_experts * k / num_tokens


def packed_expert_param_count(module: PackedExpertChoiceMoEFFN) -> int:
    return (
        module.hidden_size * module.mlp_width
        + module.mlp_width
        + module.mlp_width * module.hidden_size
        + module.hidden_size
    )


def linear_macs(module: torch.nn.Module, count: int = 1) -> int:
    if isinstance(module, torch.nn.Linear):
        return count * module.in_features * module.out_features
    return 0


def conv_patch_macs(module: torch.nn.Conv2d, num_patches: int) -> int:
    patch_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
    return num_patches * patch_dim * module.out_channels


def time_embed_macs(module: torch.nn.Sequential) -> int:
    return sum(linear_macs(child) for child in module)


def block_macs(block: DiTBlock, num_tokens: int) -> int:
    h = block.hidden_size
    a = block.attention.attention_width
    macs = 0
    macs += 6 * h * h
    macs += num_tokens * (3 * h * a + a * h) + 2 * num_tokens * num_tokens * a
    ffn = block.ffn
    if isinstance(ffn, DenseFFN):
        macs += 2 * num_tokens * h * ffn.mlp_width
    elif isinstance(ffn, (ExpertChoiceMoEFFN, PackedExpertChoiceMoEFFN)):
        k = ffn._tokens_per_expert(num_tokens)
        macs += num_tokens * h * ffn.num_experts
        macs += ffn.num_experts * k * (2 * h * ffn.mlp_width)
    else:
        raise TypeError(f"Unsupported FFN type: {type(ffn)!r}")
    return macs


def stack_macs(blocks: torch.nn.ModuleList, num_tokens: int) -> int:
    return sum(block_macs(block, num_tokens) for block in blocks)


def deferred_macs(model: ClassCondDeferredMaskingDiT, backbone_tokens: int) -> int:
    num_patches = model.num_patches
    macs = 0
    macs += conv_patch_macs(model.patch_embed, num_patches)
    macs += time_embed_macs(model.time_embed)
    macs += linear_macs(model.class_proj)
    if model.patch_mixer is not None:
        macs += stack_macs(model.patch_mixer.blocks, num_patches)
    macs += linear_macs(model.backbone_cond_proj)
    if isinstance(model.token_proj, torch.nn.Linear):
        macs += linear_macs(model.token_proj, backbone_tokens)
    macs += stack_macs(model.backbone.blocks, backbone_tokens)
    macs += 2 * model.hidden_size * model.hidden_size
    macs += backbone_tokens * model.hidden_size * model.patch_dim
    return macs


def classcond_dit_macs(model: ClassCondDiT) -> int:
    num_tokens = model.num_patches
    macs = 0
    macs += conv_patch_macs(model.patch_embed, num_tokens)
    macs += time_embed_macs(model.time_embed)
    macs += linear_macs(model.class_proj)
    macs += stack_macs(model.backbone.blocks, num_tokens)
    macs += 2 * model.hidden_size * model.hidden_size
    macs += num_tokens * model.hidden_size * model.patch_dim
    return macs


def estimate_macs(model: torch.nn.Module) -> tuple[int, int]:
    if isinstance(model, ClassCondDeferredMaskingDiT):
        train_tokens = model.num_patches
        if model.masker is not None:
            train_tokens = max(1, int(model.num_patches * (1.0 - model.masker.mask_ratio)))
        return deferred_macs(model, train_tokens), deferred_macs(model, model.num_patches)
    if isinstance(model, ClassCondDiT):
        macs = classcond_dit_macs(model)
        return macs, macs
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def b2_moe_scaling_rows() -> list[dict[str, Any]]:
    experiment = "imagenet256_latent_dit_b2_moe_layerwise"
    base = compose_model_config(experiment)
    variants = [
        mutate_moe_config(base, name="B2 layerwise dense peer", dense_peer=True),
        mutate_moe_config(base, name="B2 MoE current c2 width1"),
        mutate_moe_config(base, name="B2 MoE same total as current c1 width1", expert_capacity=1.0),
        mutate_moe_config(base, name="B2 MoE active dense c2 width0.5", moe_width_scale=0.5, expert_capacity=2.0),
        mutate_moe_config(base, name="B2 MoE active dense c1 width1", expert_capacity=1.0),
        mutate_moe_config(base, name="B2 MoE reduced active c1 width0.5", moe_width_scale=0.5, expert_capacity=1.0),
        mutate_moe_config(base, name="B2 MoE same total dense c2 width0.125", moe_width_scale=0.125, expert_capacity=2.0),
        mutate_moe_config(base, name="B2 MoE same total dense c1 width0.125", moe_width_scale=0.125, expert_capacity=1.0),
        mutate_moe_config(base, name="B2 MoE current c2 width1 PM dense", expert_capacity=2.0, patch_mixer_dense=True),
        mutate_moe_config(base, name="B2 MoE c1 width1 PM dense", expert_capacity=1.0, patch_mixer_dense=True),
        mutate_moe_config(base, name="B2 MoE c2 width0.5 PM dense", moe_width_scale=0.5, expert_capacity=2.0, patch_mixer_dense=True),
        mutate_moe_config(base, name="B2 MoE c1 width0.5 PM dense", moe_width_scale=0.5, expert_capacity=1.0, patch_mixer_dense=True),
        mutate_moe_config(
            base,
            name="B2 D8 c2 expert width0.5",
            moe_width_scale=0.5,
            expert_capacity=2.0,
            backbone_depth=8,
        ),
        mutate_moe_config(
            base,
            name="B2 D8 c2 all ffn width0.5",
            ffn_width_scale=0.5,
            expert_capacity=2.0,
            backbone_depth=8,
        ),
        mutate_moe_config(
            base,
            name="B2 D8 c1 all ffn width1",
            expert_capacity=1.0,
            backbone_depth=8,
        ),
        mutate_moe_config(
            base,
            name="B2 D8 c1 all ffn width0.5",
            ffn_width_scale=0.5,
            expert_capacity=1.0,
            backbone_depth=8,
        ),
    ]
    return [estimate(name, experiment, cfg).row() for name, cfg in variants]


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Variant | Total params | Active params | Train MACs/img | Eval MACs/img | Train FLOPs/img |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {name} | {total:.1f}M | {active:.1f}M | {train_macs:.2f}G | {eval_macs:.2f}G | {train_flops:.2f}G |".format(
                name=row["name"],
                total=row["total_params_m"],
                active=row["active_params_m"],
                train_macs=row["train_macs_g"],
                eval_macs=row["eval_macs_g"],
                train_flops=row["train_flops_g"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--preset", choices=["b2_moe_scaling"], default=None)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()
    if args.preset == "b2_moe_scaling":
        rows = b2_moe_scaling_rows()
    elif args.experiment is not None:
        cfg = compose_model_config(args.experiment)
        rows = [estimate(args.experiment, args.experiment, cfg).row()]
    else:
        parser.error("pass --experiment or --preset")
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
