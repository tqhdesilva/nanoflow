"""Structured config schema for NanoFlow. Validated by Hydra at startup."""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Dataset group


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


# Model group


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


# Flow group


@dataclass
class FlowConfig:
    _target_: str = MISSING


@dataclass
class CondOTConfig(FlowConfig):
    _target_: str = "flow.CondOT"


# Training group


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


# Sampling / inference


@dataclass
class SampleLoggerConfig:
    num_steps: int = 100
    latent_shape: Optional[list[int]] = None
    n_samples: int = 64
    guidance_scale: float = 1.0
    # TODO should we also add optional class sampler?


# Unit configs


@dataclass
class NanoFlowConfig:
    """Config for the training Trainer."""

    _target_: str = "train.Trainer"
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: Optional[str] = None


@dataclass
class DataLoaderConfig:
    _target_: str = "datasets.build_dataloader"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 128
    num_workers: int = 0
    train: bool = True


@dataclass
class InferenceUnitConfig:
    """Config for the inference FlowSampler."""

    _target_: str = "inference.FlowSampler"
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
    class_names: Optional[list[str]] = None


# Top-level


@dataclass
class InferenceConfig:
    sampler: InferenceUnitConfig
    n_samples: int = 64
    save_path: Optional[str] = None
    class_sampler: Optional[ClassSampler] = None
    metrics: Optional[list] = None


@dataclass
class Config:
    # Shared config groups, referenced by recipes via interpolation.
    trainer: Optional[NanoFlowConfig] = None
    train_loader: Optional[DataLoaderConfig] = None
    val_loader: Optional[DataLoaderConfig] = None
    inference: Optional[InferenceConfig] = None
    sample_logger: Optional[SampleLoggerConfig] = None
    device: str = "cpu"
    distributed: Optional[str] = None  # null | ddp | fsdp
    runs_dir: str = "runs"


# RL / Flow-GRPO


@dataclass
class SamplerConfig:
    T_rollout: int = 10
    sigma_a: float = 0.7
    t_min: float = 1e-3
    t_max: float = 0.999
    guidance_scale: float = 2.0


@dataclass
class RLTrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    G: int = 8
    num_inner: int = 4
    lr: float = 1e-5
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    T_inference: int = 40
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    save_every: int = 10
    log_every: int = 1
    run_prefix: str = "fashion_grpo"
    ema_decay: float = 0.0
    latent_shape: list = field(default_factory=lambda: [1, 28, 28])
    num_classes: int = 10


@dataclass
class RewardConfig:
    _target_: str = MISSING


@dataclass
class TargetClassRewardConfig(RewardConfig):
    _target_: str = "rl.reward.TargetClassReward"
    classifier_checkpoint: str = MISSING
    device: str = "mps"


@dataclass
class RolloutClientConfig:
    _target_: str = MISSING


@dataclass
class InProcessRolloutClientConfig(RolloutClientConfig):
    _target_: str = "rl.rollout_client.InProcessRolloutClient"
    device: str = "mps"
    latent_shape: list = field(default_factory=lambda: [1, 28, 28])
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass
class GRPOConfig:
    rl_training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    reward: Any = MISSING
    rollout_client: Any = MISSING
    model: Any = MISSING
    seed_checkpoint: str = MISSING
    device: str = "mps"
    distributed: Optional[str] = None
    runs_dir: str = "runs"


def _register() -> None:
    cs = ConfigStore.instance()
    # Config not registered as top-level schema: dataset/model/flow/training.
    # live as top-level Hydra groups referenced via interpolation by the recipes.
    cs.store(name="trainer_schema", node=NanoFlowConfig)
    cs.store(name="sampler_schema", node=InferenceUnitConfig)
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
    cs.store(group="rl_training", name="default_schema", node=RLTrainingConfig)
    cs.store(
        group="reward", name="fashion_classifier_schema",
        node=TargetClassRewardConfig,
    )
    cs.store(
        group="rollout_client", name="in_process_schema",
        node=InProcessRolloutClientConfig,
    )


_register()
