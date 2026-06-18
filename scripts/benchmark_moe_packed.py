#!/usr/bin/env python3
"""Benchmark loop and packed expert-choice MoE FFNs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models_dit import ExpertChoiceMoEFFN, PackedExpertChoiceMoEFFN


@dataclass(frozen=True)
class BenchShape:
    name: str
    batch_size: int
    num_tokens: int
    hidden_size: int
    num_experts: int = 8
    expert_capacity: float = 2.0
    mlp_width: int | None = None

    @property
    def width(self) -> int:
        return self.mlp_width if self.mlp_width is not None else 4 * self.hidden_size

    def verification_shape(self) -> "BenchShape":
        return BenchShape(
            name=f"{self.name}_verify",
            batch_size=min(2, self.batch_size),
            num_tokens=self.num_tokens,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_capacity=self.expert_capacity,
            mlp_width=self.mlp_width,
        )


DEFAULT_SHAPES = [
    BenchShape("patch_mixer_b8", batch_size=8, num_tokens=256, hidden_size=768),
    BenchShape("patch_mixer_b32", batch_size=32, num_tokens=256, hidden_size=768),
    BenchShape("backbone_b2_b8", batch_size=8, num_tokens=64, hidden_size=768),
    BenchShape("backbone_b2_b32", batch_size=32, num_tokens=64, hidden_size=768),
    BenchShape("backbone_m2_b8", batch_size=8, num_tokens=64, hidden_size=1024),
    BenchShape("backbone_m2_b32", batch_size=32, num_tokens=64, hidden_size=1024),
]
SMALL_VERIFY_SHAPE = BenchShape(
    "small_verify",
    batch_size=3,
    num_tokens=9,
    hidden_size=8,
    num_experts=4,
    expert_capacity=2.0,
    mlp_width=16,
)


def copy_loop_moe_to_packed(
    loop_moe: ExpertChoiceMoEFFN,
    packed_moe: PackedExpertChoiceMoEFFN,
) -> None:
    with torch.no_grad():
        packed_moe.router.weight.copy_(loop_moe.router.weight)
        for expert_idx, expert in enumerate(loop_moe.experts):
            packed_moe.w1[expert_idx].copy_(expert.net[0].weight.T)
            packed_moe.b1[expert_idx].copy_(expert.net[0].bias)
            packed_moe.w2[expert_idx].copy_(expert.net[2].weight.T)
            packed_moe.b2[expert_idx].copy_(expert.net[2].bias)


def make_pair(
    shape: BenchShape, device: torch.device
) -> tuple[ExpertChoiceMoEFFN, PackedExpertChoiceMoEFFN]:
    loop_moe = ExpertChoiceMoEFFN(
        hidden_size=shape.hidden_size,
        mlp_width=shape.width,
        num_experts=shape.num_experts,
        expert_capacity=shape.expert_capacity,
        activation=torch.nn.SiLU(),
    ).to(device)
    packed_moe = PackedExpertChoiceMoEFFN(
        hidden_size=shape.hidden_size,
        mlp_width=shape.width,
        num_experts=shape.num_experts,
        expert_capacity=shape.expert_capacity,
        activation=torch.nn.SiLU(),
    ).to(device)
    copy_loop_moe_to_packed(loop_moe, packed_moe)
    return loop_moe, packed_moe


@contextlib.contextmanager
def fp32_verification_mode(device: torch.device) -> Iterator[None]:
    if device.type != "cuda":
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def verify_pair(
    shape: BenchShape,
    device: torch.device,
    *,
    check_param_grads: bool,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    torch.manual_seed(20260617)
    with fp32_verification_mode(device):
        loop_moe, packed_moe = make_pair(shape, device)
        x = torch.randn(
            shape.batch_size,
            shape.num_tokens,
            shape.hidden_size,
            device=device,
        )
        x_loop = x.detach().clone().requires_grad_(True)
        x_packed = x.detach().clone().requires_grad_(True)
        y_loop = loop_moe(x_loop)
        y_packed = packed_moe(x_packed)
        torch.testing.assert_close(y_packed, y_loop, rtol=rtol, atol=atol)
        grad_output = torch.randn_like(y_loop)
        (y_loop * grad_output).sum().backward()
        (y_packed * grad_output).sum().backward()
        torch.testing.assert_close(x_packed.grad, x_loop.grad, rtol=rtol, atol=atol)
        if check_param_grads:
            torch.testing.assert_close(
                packed_moe.router.weight.grad,
                loop_moe.router.weight.grad,
                rtol=rtol,
                atol=atol,
            )
            for expert_idx, expert in enumerate(loop_moe.experts):
                torch.testing.assert_close(
                    packed_moe.w1.grad[expert_idx],
                    expert.net[0].weight.grad.T,
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    packed_moe.b1.grad[expert_idx],
                    expert.net[0].bias.grad,
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    packed_moe.w2.grad[expert_idx],
                    expert.net[2].weight.grad.T,
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    packed_moe.b2.grad[expert_idx],
                    expert.net[2].bias.grad,
                    rtol=rtol,
                    atol=atol,
                )
        return {
            "shape": shape.name,
            "device": device.type,
            "batch_size": shape.batch_size,
            "num_tokens": shape.num_tokens,
            "hidden_size": shape.hidden_size,
            "mlp_width": shape.width,
            "num_experts": shape.num_experts,
            "tokens_per_expert": loop_moe._tokens_per_expert(shape.num_tokens),
            "checked_param_grads": check_param_grads,
            "status": "passed",
        }


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return contextlib.nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def maybe_compile(module: torch.nn.Module, enabled: bool, mode: str | None):
    if not enabled:
        return module
    kwargs = {}
    if mode:
        kwargs["mode"] = mode
    return torch.compile(module, **kwargs)


def measure_call(
    fn, *, warmups: int, iters: int, device: torch.device
) -> tuple[float, float | None]:
    for _ in range(warmups):
        fn()
    sync_device(device)
    reset_peak_memory(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_device(device)
    elapsed = time.perf_counter() - start
    return elapsed / iters, peak_memory_mb(device)


def benchmark_module(
    module: torch.nn.Module,
    shape: BenchShape,
    device: torch.device,
    *,
    dtype_name: str,
    compiled: bool,
    compile_mode: str | None,
    warmups: int,
    iters: int,
) -> dict[str, object]:
    module.train()
    target = maybe_compile(module, compiled, compile_mode)

    def forward_once() -> None:
        with torch.no_grad(), autocast_context(device, dtype_name):
            y = target(x_forward)
        if y.numel() == 0:
            raise RuntimeError("empty output")

    def train_once() -> None:
        module.zero_grad(set_to_none=True)
        x_train.grad = None
        with autocast_context(device, dtype_name):
            y = target(x_train)
            loss = y.square().mean()
        loss.backward()

    x_forward = torch.randn(
        shape.batch_size,
        shape.num_tokens,
        shape.hidden_size,
        device=device,
    )
    x_train = x_forward.detach().clone().requires_grad_(True)
    forward_s, forward_peak = measure_call(
        forward_once,
        warmups=warmups,
        iters=iters,
        device=device,
    )
    train_s, train_peak = measure_call(
        train_once,
        warmups=warmups,
        iters=iters,
        device=device,
    )
    tokens = shape.batch_size * shape.num_tokens
    return {
        **asdict(shape),
        "mlp_width": shape.width,
        "tokens_per_expert": module._tokens_per_expert(shape.num_tokens),
        "compiled": compiled,
        "dtype": dtype_name,
        "forward_ms": forward_s * 1000.0,
        "train_step_ms": train_s * 1000.0,
        "forward_tokens_per_sec": tokens / forward_s,
        "train_tokens_per_sec": tokens / train_s,
        "forward_peak_memory_mb": forward_peak,
        "train_peak_memory_mb": train_peak,
    }


def benchmark_shape(
    shape: BenchShape,
    device: torch.device,
    *,
    dtype_name: str,
    compiled: bool,
    compile_mode: str | None,
    warmups: int,
    iters: int,
) -> list[dict[str, object]]:
    torch.manual_seed(1000 + shape.batch_size + shape.hidden_size + int(compiled))
    loop_moe, packed_moe = make_pair(shape, device)
    rows = []
    for impl, module in (("loop", loop_moe), ("packed", packed_moe)):
        row = benchmark_module(
            module,
            shape,
            device,
            dtype_name=dtype_name,
            compiled=compiled,
            compile_mode=compile_mode,
            warmups=warmups,
            iters=iters,
        )
        row["impl"] = impl
        rows.append(row)
    return rows


def speedup_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    grouped: dict[tuple[str, bool], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["name"]), bool(row["compiled"]))
        grouped.setdefault(key, {})[str(row["impl"])] = row
    for (name, compiled), pair in sorted(grouped.items()):
        if "loop" not in pair or "packed" not in pair:
            continue
        loop = pair["loop"]
        packed = pair["packed"]
        out.append(
            {
                "name": name,
                "compiled": compiled,
                "forward_speedup": float(loop["forward_ms"])
                / float(packed["forward_ms"]),
                "train_step_speedup": float(loop["train_step_ms"])
                / float(packed["train_step_ms"]),
                "loop_train_step_ms": loop["train_step_ms"],
                "packed_train_step_ms": packed["train_step_ms"],
            }
        )
    return out


def print_markdown(
    rows: list[dict[str, object]], speedups: list[dict[str, object]]
) -> None:
    print("| Shape | Impl | Compiled | Fwd ms | Train ms | Train tok/s | Peak MB |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        peak = row["train_peak_memory_mb"]
        peak_text = "" if peak is None else f"{float(peak):.1f}"
        print(
            "| {name} | {impl} | {compiled} | {fwd:.3f} | {train:.3f} | {tps:.0f} | {peak} |".format(
                name=row["name"],
                impl=row["impl"],
                compiled=row["compiled"],
                fwd=float(row["forward_ms"]),
                train=float(row["train_step_ms"]),
                tps=float(row["train_tokens_per_sec"]),
                peak=peak_text,
            )
        )
    print()
    print("| Shape | Compiled | Fwd speedup | Train speedup |")
    print("| --- | ---: | ---: | ---: |")
    for row in speedups:
        print(
            "| {name} | {compiled} | {fwd:.3f} | {train:.3f} |".format(
                name=row["name"],
                compiled=row["compiled"],
                fwd=float(row["forward_speedup"]),
                train=float(row["train_step_speedup"]),
            )
        )


def parse_shape_filter(value: str | None) -> set[str] | None:
    if value is None or value.strip() == "":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def selected_shapes(shape_filter: set[str] | None) -> list[BenchShape]:
    if shape_filter is None:
        return DEFAULT_SHAPES
    shapes = [shape for shape in DEFAULT_SHAPES if shape.name in shape_filter]
    missing = shape_filter - {shape.name for shape in shapes}
    if missing:
        raise ValueError(f"unknown shapes: {sorted(missing)}")
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true", dest="run_compile")
    parser.add_argument("--no-eager", action="store_true")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--shape-filter", default=None)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="high"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        device = torch.device("cuda", torch.cuda.current_device())
    shapes = selected_shapes(parse_shape_filter(args.shape_filter))
    compiled_modes = [] if args.no_eager else [False]
    if args.run_compile:
        compiled_modes.append(True)
    if not compiled_modes:
        parser.error("nothing to run, remove --no-eager or pass --compile")

    verification = []
    if not args.skip_verify:
        verification.append(
            verify_pair(
                SMALL_VERIFY_SHAPE,
                device,
                check_param_grads=True,
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        for shape in shapes:
            verification.append(
                verify_pair(
                    shape.verification_shape(),
                    device,
                    check_param_grads=False,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
        print(json.dumps({"verification": verification}, indent=2, sort_keys=True))

    rows = []
    for compiled in compiled_modes:
        for shape in shapes:
            rows.extend(
                benchmark_shape(
                    shape,
                    device,
                    dtype_name=args.dtype,
                    compiled=compiled,
                    compile_mode=args.compile_mode,
                    warmups=args.warmups,
                    iters=args.iters,
                )
            )
    speedups = speedup_rows(rows)
    result = {
        "device": str(device),
        "dtype": args.dtype,
        "warmups": args.warmups,
        "iters": args.iters,
        "matmul_precision": args.matmul_precision,
        "verification": verification,
        "rows": rows,
        "speedups": speedups,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        table = io.StringIO()
        with contextlib.redirect_stdout(table):
            print_markdown(rows, speedups)
        lines = [
            "# Packed MoE benchmark",
            "",
            f"device: `{device}`",
            f"dtype: `{args.dtype}`",
            f"warmups: `{args.warmups}`",
            f"iters: `{args.iters}`",
            "",
            table.getvalue().rstrip(),
        ]
        args.markdown_output.write_text("\n".join(lines).rstrip() + "\n")
    print_markdown(rows, speedups)


if __name__ == "__main__":
    main()
