"""QLoRA supervised fine-tuning for Text2SQL.

Primary path uses **Unsloth** (2x faster, ~50% less VRAM — fits Qwen2.5-Coder-1.5B
QLoRA comfortably on a free Colab T4). If Unsloth is unavailable it falls back to
plain transformers + PEFT + bitsandbytes + TRL.

Usage (Colab or local GPU):
    python -m src.train --preset exp1_qwen1.5b_bird_qlora \
        --train_file data/processed/bird_train.jsonl \
        --val_file   data/processed/bird_dev.jsonl

Quick smoke run (tiny, just to confirm the loop turns over):
    python -m src.train --max_steps 10 --train_file data/processed/bird_train.jsonl
"""
from __future__ import annotations

import argparse
import os

from .config import ExperimentConfig, PRESETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune a <=3B model for Text2SQL.")
    p.add_argument("--preset", default="exp1_qwen1.5b_bird_qlora", choices=list(PRESETS))
    p.add_argument("--base_model", default=None)
    p.add_argument("--train_file", default=None)
    p.add_argument("--val_file", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--no_4bit", action="store_true", help="LoRA in 16-bit instead of QLoRA")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = PRESETS[args.preset]
    if args.base_model: cfg.model.base_model = args.base_model
    if args.max_seq_length: cfg.model.max_seq_length = args.max_seq_length
    if args.no_4bit: cfg.model.load_in_4bit = False
    if args.train_file: cfg.data.train_file = args.train_file
    if args.val_file: cfg.data.val_file = args.val_file
    if args.output_dir: cfg.train.output_dir = args.output_dir
    if args.epochs is not None: cfg.train.num_train_epochs = args.epochs
    if args.max_steps is not None: cfg.train.max_steps = args.max_steps
    if args.lr is not None: cfg.train.learning_rate = args.lr
    if args.batch_size: cfg.train.per_device_train_batch_size = args.batch_size
    if args.grad_accum: cfg.train.gradient_accumulation_steps = args.grad_accum
    if args.lora_r: cfg.lora.r = args.lora_r
    if args.max_train_samples is not None: cfg.data.max_train_samples = args.max_train_samples
    return cfg


# --------------------------------------------------------------------------- #
# Dataset formatting: messages -> single chat-templated `text` string.
# --------------------------------------------------------------------------- #
def formatting_func(tokenizer):
    def _fmt(examples):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
        return {"text": texts}
    return _fmt


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print(f"[train] experiment = {cfg.name}")
    print(f"[train] base model = {cfg.model.base_model}  (4bit={cfg.model.load_in_4bit})")

    from datasets import load_dataset
    train_ds = load_dataset("json", data_files=cfg.data.train_file, split="train")
    if cfg.data.max_train_samples:
        train_ds = train_ds.select(range(min(cfg.data.max_train_samples, len(train_ds))))
    print(f"[train] train examples = {len(train_ds)}")

    # ----- load model (Unsloth preferred) -------------------------------- #
    use_unsloth = True
    try:
        from unsloth import FastLanguageModel
    except Exception as e:  # pragma: no cover - depends on env
        print(f"[train] Unsloth not available ({e}); using transformers+peft fallback.")
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.base_model,
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=cfg.model.load_in_4bit,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            target_modules=cfg.lora.target_modules,
            use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
            random_state=cfg.lora.random_state,
        )
    else:
        model, tokenizer = _load_hf_peft(cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(formatting_func(tokenizer), batched=True,
                            remove_columns=train_ds.column_names)

    # ----- trainer ------------------------------------------------------- #
    from trl import SFTTrainer, SFTConfig
    sft_args = SFTConfig(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        seed=cfg.train.seed,
        bf16=_bf16_supported(),
        fp16=not _bf16_supported(),
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
    )

    # Completion-only loss: mask everything before the assistant turn so the
    # model is graded only on the SQL it must produce, not on the schema/question.
    if cfg.train.train_on_completion_only and use_unsloth:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.train.output_dir, "experiment_config.json"))

    print("[train] starting training ...")
    trainer.train()

    print(f"[train] saving adapter + tokenizer -> {cfg.train.output_dir}")
    model.save_pretrained(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    print("[train] done.")


def _bf16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _load_hf_peft(cfg: ExperimentConfig):
    """Fallback loader without Unsloth (plain transformers + peft + bitsandbytes)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = None
    if cfg.model.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
    )
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    peft_cfg = PeftLoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.lora_alpha, lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias, target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == "__main__":
    main()
