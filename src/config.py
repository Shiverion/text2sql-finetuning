"""Central configuration for the Text2SQL fine-tuning pipeline.

Everything that an experiment might want to vary lives here as a dataclass with
sane defaults, so a single ``ExperimentConfig`` object fully describes a run.
The training / inference scripts also expose CLI flags that override these.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # Base checkpoint. Unsloth ships pre-quantized mirrors that load faster on
    # the free Colab T4 ("unsloth/Qwen2.5-Coder-1.5B-Instruct"). The plain HF id
    # ("Qwen/Qwen2.5-Coder-1.5B-Instruct") also works.
    base_model: str = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # QLoRA. Set False for LoRA in fp16/bf16.
    dtype: Optional[str] = None        # None -> auto (bf16 on Ampere+, else fp16)


# --------------------------------------------------------------------------- #
# LoRA / PEFT
# --------------------------------------------------------------------------- #
@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0          # 0.0 is Unsloth-optimized (no recompute)
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    output_dir: str = "outputs/qwen2.5-coder-1.5b-bird-qlora"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2     # effective batch = 16
    warmup_ratio: float = 0.03
    num_train_epochs: float = 2.0
    max_steps: int = -1                       # >0 overrides epochs (quick runs)
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    logging_steps: int = 20
    save_steps: int = 500
    seed: int = 3407
    # Mask the prompt tokens so loss is computed only on the SQL completion.
    train_on_completion_only: bool = True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    train_file: str = "data/processed/bird_train.jsonl"
    val_file: str = "data/processed/bird_dev.jsonl"
    dialect: str = "SQLite"
    include_evidence: bool = True       # BIRD ships an "evidence" knowledge hint
    schema_mode: str = "ddl"            # "ddl" (CREATE TABLE) or "compact"
    max_train_samples: Optional[int] = None   # subsample for fast experiments


# --------------------------------------------------------------------------- #
# Top-level experiment bundle
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "exp1_qwen1.5b_bird_qlora"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(
            name=raw.get("name", "exp"),
            model=ModelConfig(**raw["model"]),
            lora=LoraConfig(**raw["lora"]),
            train=TrainConfig(**raw["train"]),
            data=DataConfig(**raw["data"]),
        )


# Convenience presets referenced in the report's experiment table.
PRESETS = {
    # Exp 1 — main run: 1.5B coder model, BIRD, QLoRA.
    "exp1_qwen1.5b_bird_qlora": ExperimentConfig(name="exp1_qwen1.5b_bird_qlora"),

    # Exp 2 — "smaller is better" ablation: 0.5B model, same data/recipe.
    "exp2_qwen0.5b_bird_qlora": ExperimentConfig(
        name="exp2_qwen0.5b_bird_qlora",
        model=ModelConfig(base_model="unsloth/Qwen2.5-Coder-0.5B-Instruct"),
        train=TrainConfig(
            output_dir="outputs/qwen2.5-coder-0.5b-bird-qlora",
            per_device_train_batch_size=16,
        ),
    ),

    # Exp 3 — data-scaling: BIRD train + a SynSQL-2.5M subset.
    "exp3_qwen1.5b_bird_plus_synsql": ExperimentConfig(
        name="exp3_qwen1.5b_bird_plus_synsql",
        train=TrainConfig(
            output_dir="outputs/qwen2.5-coder-1.5b-bird-synsql-qlora",
            num_train_epochs=1.0,
        ),
        data=DataConfig(train_file="data/processed/bird_plus_synsql_train.jsonl"),
    ),
}
