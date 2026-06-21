"""Generate SQL predictions for a JSONL of Text2SQL records.

Reads the standardized records produced by ``data_prep`` (or any JSONL with
``schema`` / ``question`` / ``evidence`` fields), runs the fine-tuned model, and
writes a predictions JSONL that ``evaluate`` consumes.

Usage:
    python -m src.inference \
        --model_dir outputs/qwen2.5-coder-1.5b-bird-qlora \
        --input data/processed/bird_dev.jsonl \
        --output outputs/preds_bird_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .prompts import build_messages, extract_sql


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Text2SQL inference.")
    p.add_argument("--model_dir", required=True,
                   help="adapter dir (fine-tuned) OR a base model id for zero-shot baseline")
    p.add_argument("--input", required=True, help="JSONL of records")
    p.add_argument("--output", required=True, help="JSONL of predictions")
    p.add_argument("--base_model", default=None,
                   help="base model id if --model_dir holds only an adapter without config")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--dialect", default="SQLite")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_4bit", action="store_true")
    return p.parse_args()


def load_records(path: str, limit=None) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def load_model(model_dir: str, max_seq_length: int, load_in_4bit: bool):
    """Load a fine-tuned adapter (or plain base model) for inference."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except Exception as e:
        print(f"[inference] Unsloth path unavailable ({e}); using transformers.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()
        return model, tokenizer, "hf"


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int) -> List[str]:
    import torch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # greedy: deterministic, best for SQL
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, decode only the newly generated continuation.
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    records = load_records(args.input, args.limit)
    print(f"[inference] {len(records)} records from {args.input}")

    model, tokenizer, backend = load_model(
        args.model_dir, args.max_seq_length, load_in_4bit=not args.no_4bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # required for correct decoder generation

    preds: List[Dict] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(r["schema"], r["question"], r.get("evidence", ""), args.dialect),
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        raw_outs = generate_batch(model, tokenizer, prompts, args.max_new_tokens)
        for r, raw in zip(batch, raw_outs):
            preds.append({
                "db_id": r["db_id"],
                "question": r["question"],
                "gold_sql": r.get("sql", ""),
                "pred_sql": extract_sql(raw),
                "raw_output": raw,
                "db_path": r.get("db_path", ""),
                "difficulty": r.get("difficulty", ""),
            })
        print(f"[inference] {min(start + args.batch_size, len(records))}/{len(records)}")

    with open(args.output, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[inference] wrote {len(preds)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
