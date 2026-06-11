"""DiT-style transformer models for latent flow matching."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models import SinusoidalEmbedding


class RoPE2D(nn.Module):
    """Fixed 2D rotary position embedding for attention Q and K."""

    def __init__(self, base: float = 10000.0):
        super().__init__()
        self.base = base

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_2d_rope(q, k, coords, base=self.base)


def build_2d_patch_coords(grid_h: int, grid_w: int, device=None) -> torch.Tensor:
    """Return row-major patch coordinates as an int64 tensor of shape [N, 2]."""
    rows, cols = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    return torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1).long()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_1d_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    dim = x.size(-1)
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    angles = (
        positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None]
    )
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    shape = [1] * x.dim()
    shape[-3] = positions.numel()
    shape[-1] = dim
    cos = cos.view(*shape)
    sin = sin.view(*shape)
    return x * cos + _rotate_half(x) * sin


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    *,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply row and column RoPE to q and k.

    q and k are [B, N, H, D]. coords is [N, 2] with row, col positions.
    Half of each head is rotated by row and half by column.
    """
    head_dim = q.size(-1)
    if head_dim % 4 != 0:
        raise ValueError(f"2D RoPE requires head_dim divisible by 4, got {head_dim}")
    row_dim = head_dim // 2
    col_dim = head_dim - row_dim
    q_row, q_col = q.split([row_dim, col_dim], dim=-1)
    k_row, k_col = k.split([row_dim, col_dim], dim=-1)
    rows = coords[:, 0]
    cols = coords[:, 1]
    q = torch.cat(
        [
            _apply_1d_rope(q_row, rows, base=base),
            _apply_1d_rope(q_col, cols, base=base),
        ],
        dim=-1,
    )
    k = torch.cat(
        [
            _apply_1d_rope(k_row, rows, base=base),
            _apply_1d_rope(k_col, cols, base=base),
        ],
        dim=-1,
    )
    return q, k


class AdaLayerNorm(nn.Module):
    """AdaLN-Zero layer with owned conditioning projection."""

    def __init__(self, hidden_size: int, use_gate: bool = True, eps: float = 1e-6):
        super().__init__()
        self.use_gate = use_gate
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        n_outputs = 3 if use_gate else 2
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_outputs * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        chunks = self.modulation(cond).chunk(3 if self.use_gate else 2, dim=1)
        shift = chunks[0]
        scale = chunks[1]
        h = self.norm(x)
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        if self.use_gate:
            gate = chunks[2]
            return h, gate[:, None, :]
        return h


class SelfAttention(nn.Module):
    """Multi-head self-attention with fixed 2D RoPE on Q and K."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_width: Optional[int] = None,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_width = attention_width or hidden_size
        if self.attention_width % num_heads != 0:
            raise ValueError(
                "attention_width must be divisible by num_heads, got "
                f"{self.attention_width} and {num_heads}"
            )
        self.head_dim = self.attention_width // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"2D RoPE requires per-head dim divisible by 4, got {self.head_dim}"
            )
        self.qkv = nn.Linear(hidden_size, 3 * self.attention_width, bias=qkv_bias)
        self.proj = nn.Linear(self.attention_width, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        pos_embedding: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        bsz, num_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if pos_embedding is not None:
            if coords is None:
                raise ValueError("coords are required when pos_embedding is set")
            q, k = pos_embedding(q, k, coords.to(x.device))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = (
            out.transpose(1, 2).contiguous().view(bsz, num_tokens, self.attention_width)
        )
        return self.proj(out)


class DenseFFN(nn.Module):
    """Dense transformer feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        mlp_width: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_width = mlp_width or 4 * hidden_size
        if activation is None:
            self.activation = nn.GELU(approximate="tanh")
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            raise TypeError(
                "DenseFFN activation must be an nn.Module or None. "
                "Use Hydra recursive instantiation for activation configs."
            )
        self.net = nn.Sequential(
            nn.Linear(hidden_size, self.mlp_width),
            self.activation,
            nn.Linear(self.mlp_width, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT transformer block."""

    def __init__(self, hidden_size: int, attention: nn.Module, ffn: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_adaln = AdaLayerNorm(hidden_size)
        self.ffn_adaln = AdaLayerNorm(hidden_size)
        self.attention = attention
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        pos_embedding: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        h, attn_gate = self.attn_adaln(x, cond)
        x = x + attn_gate * self.attention(h, coords, pos_embedding)
        h, ffn_gate = self.ffn_adaln(x, cond)
        x = x + ffn_gate * self.ffn(h)
        return x


class DiTBackbone(nn.Module):
    """Ordered DiT block stack. The block list is the topology source of truth."""

    def __init__(
        self,
        hidden_size: int,
        blocks: list[nn.Module],
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not blocks:
            raise ValueError("DiTBackbone requires at least one block")
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        pos_embedding: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda h, c: block(h, c, coords, pos_embedding),
                    x,
                    cond,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, coords, pos_embedding)
        return x


class ClassCondDiT(nn.Module):
    """Class-conditioned DiT for latent flow matching."""

    def __init__(
        self,
        in_ch: int,
        latent_size: int,
        patch_size: int,
        num_classes: int,
        backbone: nn.Module,
        time_dim: Optional[int] = None,
        class_dim: Optional[int] = None,
        pos_embedding: Optional[nn.Module] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(
                f"latent_size {latent_size} must be divisible by patch_size {patch_size}"
            )
        self.in_ch = in_ch
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.null_token = num_classes
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_ch * patch_size * patch_size
        self.pos_embedding = pos_embedding or RoPE2D()

        self.backbone = backbone
        if use_gradient_checkpointing and hasattr(
            self.backbone, "use_gradient_checkpointing"
        ):
            self.backbone.use_gradient_checkpointing = True
        self.hidden_size = self.backbone.hidden_size

        time_dim = time_dim or self.hidden_size
        class_dim = class_dim or self.hidden_size
        self.patch_embed = nn.Conv2d(
            in_ch,
            self.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)
        self.class_proj = (
            nn.Identity()
            if class_dim == self.hidden_size
            else nn.Linear(class_dim, self.hidden_size)
        )
        self.final_adaln = AdaLayerNorm(self.hidden_size, use_gate=False)
        self.patch_proj = nn.Linear(self.hidden_size, self.patch_dim)
        nn.init.zeros_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        self.register_buffer(
            "patch_coords",
            build_2d_patch_coords(self.grid_size, self.grid_size),
            persistent=False,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} channels, got {x.size(1)}")
        if x.size(2) != self.latent_size or x.size(3) != self.latent_size:
            raise ValueError(
                f"Expected spatial size {self.latent_size}, got {tuple(x.shape[-2:])}"
            )
        h = self.patch_embed(x)
        return h.flatten(2).transpose(1, 2)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 3:
            raise ValueError(
                f"Expected patches shape [B, N, D], got {tuple(patches.shape)}"
            )
        if patches.size(1) != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {patches.size(1)}"
            )
        if patches.size(2) != self.patch_dim:
            raise ValueError(
                f"Expected patch dim {self.patch_dim}, got {patches.size(2)}"
            )
        p = self.patch_size
        h = self.grid_size
        w = self.grid_size
        x = patches.view(patches.size(0), h, w, self.in_ch, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(patches.size(0), self.in_ch, h * p, w * p)

    def _condition(
        self, x: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor]
    ):
        if labels is None:
            labels = torch.full(
                (x.size(0),),
                self.null_token,
                device=x.device,
                dtype=torch.long,
            )
        else:
            labels = labels.to(device=x.device, dtype=torch.long)
        t = t.to(device=x.device, dtype=x.dtype)
        return self.time_embed(t) + self.class_proj(self.class_embed(labels))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(x, t, labels)
        h = self.patchify(x)
        h = self.backbone(
            h,
            cond,
            self.patch_coords.to(x.device),
            self.pos_embedding,
        )
        h = self.final_adaln(h, cond)
        patches = self.patch_proj(h)
        return self.unpatchify(patches)
