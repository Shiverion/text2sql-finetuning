"""Text2SQL fine-tuning package.

Modules
-------
config        : central dataclass configuration (paths, model, LoRA, training).
schema_utils  : extract / serialize SQLite database schemas for prompting.
prompts       : build the instruction prompt and chat-message list.
data_prep     : turn BIRD / SynSQL raw data into standardized JSONL records.
train         : QLoRA supervised fine-tuning (Unsloth, with a plain-PEFT fallback).
inference     : generate SQL from (question + schema) for a JSONL of records.
evaluate      : execution accuracy / valid-SQL rate / exact match against SQLite.
"""

__all__ = [
    "config",
    "schema_utils",
    "prompts",
    "data_prep",
    "train",
    "inference",
    "evaluate",
]
