"""Structured config schema for NanoFlow — validated by Hydra at startup."""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ---- dataset group ----


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    _partial_: bool = True
    name: str = MISSING


@dataclass
class MoonsDatasetConfig(DatasetConfig):
    _target_: str = "datasets.MoonsDataset"
    n: int = 8000
    noise: float = 0.05


@dataclass
class FashionDatasetConfig(DatasetConfig):
    _target_: str = "datasets.FashionMNISTDataset"
    root: str = "./data"


@dataclass
class CifarDatasetConfig(DatasetConfig):
    _target_: str = "datasets.CifarDataset"
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


@dataclass
class ClassCondMLPConfig(ModelConfig):
    _target_: str = "models.ClassCondMLP"
    hidden_dim: int = 128
    num_layers: int = 4
    time_dim: int = 32
    num_classes: int = 2


@dataclass
class ClassCondUNetFashionConfig(ModelConfig):
    _target_: str = "models.ClassCondUNet"
    in_ch: int = 1
    base_ch: int = 32
    depth: int = 2
    time_dim: int = 64
    use_attn: bool = False
    num_classes: int = 10


@dataclass
class ClassCondUNetCifarConfig(ModelConfig):
    _target_: str = "models.ClassCondUNet"
    in_ch: int = 3
    base_ch: int = 128
    depth: int = 3
    time_dim: int = 256
    use_attn: bool = True
    num_classes: int = 10


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
    resume: Optional[str] = None
    run_prefix: str = "${dataset.name}"
    log_every: int = 50
    grad_clip: float = 1.0
    ema_decay: float = 0
    num_workers: int = 0
    precision: Optional[str] = None
    p_uncond: Optional[float] = None


# ---- sampling / inference ----


@dataclass
class SampleLoggerConfig:
    num_steps: int = 100
    latent_shape: Optional[list[int]] = None
    n_samples: int = 64
    guidance_scale: float = 1.0
    # TODO should we also add optional class sampler?


# ---- unit configs ----


@dataclass
class NanoFlowConfig:
    """Config for FlowMatchingUnit (training only)."""

    _target_: str = "unit.FlowMatchingUnit"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logger: Optional[SampleLoggerConfig] = None
    device: str = "cpu"
    runs_dir: str = "runs"


@dataclass
class InferenceUnitConfig:
    """Config for InferenceUnit."""

    _target_: str = "unit.InferenceUnit"
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: Optional[str] = None
    num_steps: int = 100
    latent_shape: Optional[list[int]] = None
    device: str = "cpu"


@dataclass
class ClassSampler:
    num_classes: int = 10
    guidance_scale: float = 1.0
    probs: Optional[list[float]] = None


# ---- top-level ----


@dataclass
class InferenceConfig:
    infer_unit: InferenceUnitConfig
    n_samples: int = 64
    save_path: Optional[str] = None
    class_sampler: Optional[ClassSampler] = None


@dataclass
class Config:
    # Shared — populated by Hydra config groups, referenced by units via interpolation
    train_unit: Optional[NanoFlowConfig] = None
    inference: Optional[InferenceConfig] = None


def _register() -> None:
    cs = ConfigStore.instance()
    # Config not registered as top-level schema — dataset/model/flow/training
    # live as top-level Hydra groups referenced via interpolation by the units.
    cs.store(name="train_unit_schema", node=NanoFlowConfig)
    cs.store(name="infer_unit_schema", node=InferenceUnitConfig)
    cs.store(group="dataset", name="moons_schema", node=MoonsDatasetConfig)
    cs.store(group="dataset", name="fashion_schema", node=FashionDatasetConfig)
    cs.store(group="dataset", name="cifar10_schema", node=CifarDatasetConfig)
    cs.store(group="model", name="mlp_schema", node=MLPConfig)
    cs.store(group="model", name="unet_fashion_schema", node=UNetFashionConfig)
    cs.store(group="model", name="unet_cifar_schema", node=UNetCifarConfig)
    cs.store(group="model", name="classcond_mlp_schema", node=ClassCondMLPConfig)
    cs.store(
        group="model",
        name="classcond_unet_fashion_schema",
        node=ClassCondUNetFashionConfig,
    )
    cs.store(
        group="model",
        name="classcond_unet_cifar_schema",
        node=ClassCondUNetCifarConfig,
    )
    cs.store(group="flow", name="condot_schema", node=CondOTConfig)
    cs.store(group="training", name="default_schema", node=TrainingConfig)


_register()
