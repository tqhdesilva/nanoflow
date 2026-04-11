"""Structured config schema for NanoFlow — validated by Hydra at startup."""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# ---- dataset group ----


@dataclass
class DatasetConfig:
    name: str = MISSING


@dataclass
class MoonsDatasetConfig(DatasetConfig):
    n: int = 8000
    noise: float = 0.05


@dataclass
class ImageDatasetConfig(DatasetConfig):
    root: str = "./data"


# ---- model group ----


@dataclass
class ModelConfig:
    _target_: str = MISSING


@dataclass
class MLPConfig(ModelConfig):
    _target_: str = "models.MLP"
    hidden_dim: int = 128
    num_layers: int = 4
    time_dim: int = 32


@dataclass
class UNetFashionConfig(ModelConfig):
    _target_: str = "models.UNet"
    in_ch: int = 1
    base_ch: int = 32
    depth: int = 2
    time_dim: int = 64
    use_attn: bool = False


@dataclass
class UNetCifarConfig(ModelConfig):
    _target_: str = "models.UNet"
    in_ch: int = 3
    base_ch: int = 128
    depth: int = 3
    time_dim: int = 256
    use_attn: bool = True


# ---- flow group ----


@dataclass
class FlowConfig:
    _target_: str = MISSING


@dataclass
class CondOTConfig(FlowConfig):
    _target_: str = "flow.CondOT"


# ---- training group ----


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    warmup_epochs: int = 0
    save_every: int = 10
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None
    run_name: str = "${dataset.name}_${now:%Y%m%d_%H%M%S}"
    log_dir: str = "runs/${training.run_name}"
    log_every: int = 50
    grad_clip: float = 1.0
    ema_decay: float = 0
    num_workers: int = 0
    precision: Optional[str] = None


# ---- inference group ----


@dataclass
class InferenceConfig:
    n_samples: int = 64
    num_steps: int = 100
    image_shape: Optional[list[int]] = None
    samples_plot: str = "${dataset.name}_samples.png"


# ---- top-level ----


@dataclass
class NanoFlowConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: str = "cpu"
    save: bool = False


def _register() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=NanoFlowConfig)
    cs.store(group="dataset", name="moons_schema", node=MoonsDatasetConfig)
    cs.store(group="dataset", name="fashion_schema", node=ImageDatasetConfig)
    cs.store(group="dataset", name="cifar10_schema", node=ImageDatasetConfig)
    cs.store(group="model", name="mlp_schema", node=MLPConfig)
    cs.store(group="model", name="unet_fashion_schema", node=UNetFashionConfig)
    cs.store(group="model", name="unet_cifar_schema", node=UNetCifarConfig)
    cs.store(group="flow", name="condot_schema", node=CondOTConfig)
    cs.store(group="training", name="default_schema", node=TrainingConfig)
    cs.store(group="inference", name="moons_schema", node=InferenceConfig)
    cs.store(group="inference", name="fashion_schema", node=InferenceConfig)
    cs.store(group="inference", name="cifar10_schema", node=InferenceConfig)


_register()
