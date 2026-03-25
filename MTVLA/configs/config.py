"""Config dataclasses and YAML loader for MTVLA."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    name: str = "MemoryTreeVLA"

    # ---- shared embedding dimension ----
    embed_dim: int = 512
    action_dim: int = 7           # per-step action dim (Δxyz Δrxryrz gripper)
    max_depth: int = 4

    # ---- Vision Mamba ----
    vision_channels: int = 96     # stem output channels (→ embed_dim via proj)
    vision_layers: int = 4
    image_size: int = 224

    # ---- Task Tree Mamba ----
    node_vocab_size: int = 512
    tree_layers: int = 2

    # ---- Multimodal Mamba ----
    mm_mamba_layers: int = 4
    mm_d_state: int = 16

    # ---- Action LLM (Qwen2.5-0.5B) ----
    action_llm_path: str = "checkpoints/Qwen2.5-0.5B"
    action_llm_dim: int = 896     # hidden size of Qwen2.5-0.5B
    action_llm_layers: int = 14   # number of transformer layers to keep
    freeze_action_llm: bool = False

    # ---- Tree LLM (Qwen2.5-1.5B-Instruct) ----
    tree_llm_path: str = "checkpoints/Qwen2.5-1.5B-Instruct"
    tree_llm_dim: int = 1536      # hidden size of Qwen2.5-1.5B
    freeze_tree_llm: bool = False

    # ---- ActionConditionBuilder ----
    state_dim: int = 15           # 7 joint pos + 7 joint vel + 1 gripper
    num_state_tokens: int = 2
    llm_seq_pool: int = 32        # keep last 32 LLM tokens as context

    # ---- Flow Matching Action Head ----
    action_horizon: int = 16
    action_head_layers: int = 8
    action_head_heads: int = 8
    num_inference_timesteps: int = 20


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    output_dir: str = "outputs/"


# ---------------------------------------------------------------------------
# Dataset config  (per benchmark)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for a single benchmark dataset."""
    name: str = "robocerebra"          # robocerebra | libero | calvin | robomme
    data_root: str = "data/"
    split: str = "train"               # train / val / test / D->D (CALVIN)
    suite: str = "all"                 # LIBERO suite or ROBOMME category
    num_workers: int = 4
    image_size: int = 224
    max_episode_steps: int = 512       # max steps per episode in loader


# ---------------------------------------------------------------------------
# Per-stage training config
# ---------------------------------------------------------------------------

@dataclass
class StageTrainConfig:
    """Hyper-parameters for one training stage."""
    # whether to run this stage
    enabled: bool = True
    # resume from a specific checkpoint *for this stage*; None = start fresh
    resume_from: Optional[str] = None

    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    # per-module learning-rate multipliers applied on top of `lr`
    # keys must match sub-module names in MemoryTreeVLA
    lr_multipliers: Dict[str, float] = field(default_factory=dict)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    # datasets to use for this stage (list of DatasetConfig dicts in YAML)
    datasets: List[DatasetConfig] = field(default_factory=list)


@dataclass
class TrainingPipelineConfig:
    """Top-level three-stage training pipeline config."""
    output_dir: str = "outputs/"
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    deepspeed_config: Optional[str] = "MTVLA/configs/ds_config.json"
    log_interval: int = 50           # log every N steps
    save_interval: int = 1000        # save checkpoint every N steps
    eval_interval: int = 500         # run eval every N steps (0 = disabled)

    # ---- Stage 1: Tree LLM + Multimodal Mamba ----
    # Dataset: RoboCerebra_trainset (fixed, not configurable)
    stage1: StageTrainConfig = field(default_factory=lambda: StageTrainConfig(
        epochs=10,
        batch_size=8,
        lr=1e-4,
        warmup_steps=500,
        datasets=[DatasetConfig(
            name="robocerebra",
            data_root="dataset/RoboCerebra/RoboCerebra_trainset",
        )],
    ))

    # ---- Stage 2: Multimodal Mamba + Action Head ----
    stage2: StageTrainConfig = field(default_factory=lambda: StageTrainConfig(
        epochs=20,
        batch_size=16,
        lr=5e-5,
        warmup_steps=300,
        datasets=[],              # filled from YAML
    ))

    # ---- Stage 3: End-to-end fine-tuning ----
    stage3: StageTrainConfig = field(default_factory=lambda: StageTrainConfig(
        epochs=10,
        batch_size=8,
        lr=1e-5,
        lr_multipliers={"action_llm": 0.1},
        warmup_steps=200,
        datasets=[],              # filled from YAML
    ))


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Legacy single-dataset config kept for backward compatibility."""
    dataset: str = "libero"
    data_root: str = "data/"
    num_workers: int = 4
    image_size: int = 224


@dataclass
class MTVLAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    # legacy flat train / data blocks (used when `pipeline` is absent)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # three-stage pipeline (preferred)
    pipeline: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_dataset_config(raw: dict) -> DatasetConfig:
    return DatasetConfig(**{k: v for k, v in raw.items()
                            if k in DatasetConfig.__dataclass_fields__})


def _load_stage_config(raw: dict) -> StageTrainConfig:
    datasets_raw = raw.pop("datasets", [])
    sc = StageTrainConfig(**{k: v for k, v in raw.items()
                             if k in StageTrainConfig.__dataclass_fields__})
    sc.datasets = [_load_dataset_config(d) for d in (datasets_raw or [])]
    return sc


def load_config(path: str) -> MTVLAConfig:
    """Load a YAML config file and return an MTVLAConfig instance."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = MTVLAConfig()
    if "model" in raw:
        cfg.model = ModelConfig(**{k: v for k, v in raw["model"].items()
                                   if k in ModelConfig.__dataclass_fields__})
    if "train" in raw:
        cfg.train = TrainConfig(**{k: v for k, v in raw["train"].items()
                                   if k in TrainConfig.__dataclass_fields__})
    if "data" in raw:
        cfg.data = DataConfig(**{k: v for k, v in raw["data"].items()
                                 if k in DataConfig.__dataclass_fields__})
    if "pipeline" in raw:
        p = raw["pipeline"]
        pc = TrainingPipelineConfig(**{k: v for k, v in p.items()
                                        if k not in ("stage1", "stage2", "stage3")
                                        and k in TrainingPipelineConfig.__dataclass_fields__})
        if "stage1" in p:
            pc.stage1 = _load_stage_config(dict(p["stage1"]))
        if "stage2" in p:
            pc.stage2 = _load_stage_config(dict(p["stage2"]))
        if "stage3" in p:
            pc.stage3 = _load_stage_config(dict(p["stage3"]))
        cfg.pipeline = pc
    return cfg
