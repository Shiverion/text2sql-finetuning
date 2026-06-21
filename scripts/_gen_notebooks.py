"""Generate the three deliverable notebooks as valid .ipynb JSON.

Run once: `python scripts/_gen_notebooks.py`. Kept in the repo so the notebooks
are reproducible/diff-able rather than hand-edited binary-ish JSON.
"""
import json, os

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write(name, cells):
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook(cells), fh, indent=1)
    print("wrote", os.path.normpath(path))


# ===================================================================== #
# 01 — Data exploration
# ===================================================================== #
nb1 = [
 md("# 01 · Dataset Exploration — BIRD & SynSQL-2.5M",
    "",
    "Goal: understand the data **before** training — size, difficulty mix, SQL",
    "complexity, schema breadth — so the fine-tuning recipe is grounded in",
    "evidence rather than guesswork.",
    "",
    "Runs on CPU. If the official BIRD files aren't present it falls back to the",
    "bundled synthetic `data/sample/` so the notebook always executes."),
 code("!pip install -q datasets pandas matplotlib"),
 code("import os, json, glob, re",
      "import pandas as pd",
      "import matplotlib.pyplot as plt",
      "FIG = '../report/figures'; os.makedirs(FIG, exist_ok=True)"),
 md("## 1. Load BIRD train (or fall back to the sample)",
    "",
    "Official BIRD download (run on Colab if you want the full set):",
    "```bash",
    "!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip && unzip -q train.zip",
    "!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip && unzip -q dev.zip",
    "```",
    "Links occasionally move — see https://bird-bench.github.io for the current ones,",
    "or use a 🤗 mirror such as `lamini/bird_text_to_sql`."),
 code("def load_examples():",
      "    for p in ['train/train.json', 'dev/dev.json', 'data/train.json']:",
      "        if os.path.exists(p):",
      "            print('using', p)",
      "            return json.load(open(p, encoding='utf-8')), p",
      "    print('BIRD not found — using bundled sample')",
      "    return json.load(open('../data/sample/examples.json', encoding='utf-8')), 'sample'",
      "",
      "rows, source = load_examples()",
      "df = pd.DataFrame(rows)",
      "df['sql'] = df.get('SQL', df.get('sql'))",
      "print('examples:', len(df))",
      "df.head(3)"),
 md("## 2. Difficulty distribution",
    "BIRD labels each dev question simple / moderate / challenging. The mix tells",
    "us where accuracy will be hardest and how to read per-bucket EX later."),
 code("if 'difficulty' in df and df['difficulty'].notna().any():",
      "    vc = df['difficulty'].value_counts()",
      "    print(vc)",
      "    ax = vc.plot(kind='bar', title='Question difficulty', rot=0)",
      "    ax.figure.tight_layout(); ax.figure.savefig(f'{FIG}/difficulty.png', dpi=120)",
      "else:",
      "    print('no difficulty labels in this split')"),
 md("## 3. SQL complexity — length & keyword frequency",
    "Proxies for how hard generation is: long queries, JOINs, aggregation, nesting."),
 code("df['sql_len'] = df['sql'].str.len()",
      "df['n_tokens'] = df['sql'].str.split().apply(len)",
      "print(df[['sql_len','n_tokens']].describe())",
      "ax = df['n_tokens'].plot(kind='hist', bins=30, title='SQL length (tokens)')",
      "ax.figure.tight_layout(); ax.figure.savefig(f'{FIG}/sql_len.png', dpi=120)"),
 code("KEYWORDS = ['JOIN','LEFT JOIN','GROUP BY','ORDER BY','WHERE','HAVING',",
      "            'DISTINCT','LIMIT','COUNT','SUM','AVG','MAX','MIN','CASE',",
      "            'SELECT.*SELECT']  # last = nested subquery (regex)",
      "up = df['sql'].str.upper()",
      "freq = {k: int(up.str.contains(k, regex=True).sum()) for k in KEYWORDS}",
      "freq = dict(sorted(freq.items(), key=lambda x: -x[1]))",
      "print(json.dumps(freq, indent=2))",
      "ax = pd.Series(freq).plot(kind='barh', title='SQL keyword frequency')",
      "ax.invert_yaxis(); ax.figure.tight_layout(); ax.figure.savefig(f'{FIG}/keywords.png', dpi=120)"),
 md("## 4. Schema breadth",
    "How many distinct databases, and how wide are they? Wide schemas are the",
    "main reason a small model fails: the relevant columns get lost in a long",
    "prompt. This motivates schema serialization choices in `src/schema_utils.py`."),
 code("if 'db_id' in df:",
      "    print('distinct databases:', df['db_id'].nunique())",
      "    print(df['db_id'].value_counts().head(10))"),
 code("# Column counts per sample DB (works for the bundled sample; on full BIRD",
      "# point glob at dev_databases/**/*.sqlite).",
      "import sqlite3",
      "dbs = glob.glob('../data/sample/db/**/*.sqlite', recursive=True) or glob.glob('dev_databases/**/*.sqlite', recursive=True)",
      "for db in dbs[:10]:",
      "    con = sqlite3.connect(db)",
      "    tabs = [r[0] for r in con.execute(\"select name from sqlite_master where type='table'\")]",
      "    ncols = sum(len(con.execute(f'PRAGMA table_info(\"{t}\")').fetchall()) for t in tabs)",
      "    print(os.path.basename(db), '->', len(tabs), 'tables,', ncols, 'columns')",
      "    con.close()"),
 md("## 5. Takeaways → recipe",
    "",
    "- **Schema is the prompt's bulk** → serialize compactly; keep `max_seq_length`",
    "  large enough (2048) to fit the biggest schemas without truncation.",
    "- **Evidence/hints matter** (BIRD) → always include them in the prompt.",
    "- **JOIN + GROUP BY dominate** → these are the patterns LoRA must learn; the",
    "  fintech use-case ('top merchants last quarter') is exactly this shape.",
    "- **Difficulty is skewed to simple/moderate** → expect most EX gains there;",
    "  challenging (nested, multi-join) will lag — call this out in the report."),
]

# ===================================================================== #
# 02 — Fine-tuning (Colab)
# ===================================================================== #
nb2 = [
 md("# 02 · Fine-tune Qwen2.5-Coder (QLoRA) for Text2SQL — Colab T4",
    "",
    "Free-tier recipe: **Unsloth + QLoRA** on `Qwen2.5-Coder-1.5B-Instruct`",
    "(Apache-2.0, 1.54B params, code-specialised). ~1.5–2 GB VRAM for the adapter;",
    "fits a free T4 with room to spare.",
    "",
    "> Runtime → Change runtime type → **T4 GPU** before running."),
 code("import torch; print('CUDA:', torch.cuda.is_available(),",
      "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"),
 md("## 1. Install"),
 code("!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
      "!pip install -q --no-deps trl peft accelerate bitsandbytes datasets"),
 md("## 2. Get the code + data",
    "Clone your repo (or upload `src/`). Then download BIRD."),
 code("# !git clone https://github.com/<you>/text2sql-finetuning.git && cd text2sql-finetuning",
      "import sys; sys.path.insert(0, '.')  # ensure `src` importable"),
 code("# --- BIRD (official). Skip if you uploaded your own processed JSONL. ---",
      "!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip && unzip -q -o train.zip",
      "!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip   && unzip -q -o dev.zip",
      "# BIRD nests databases one level deep; adjust paths if the archive layout differs."),
 md("## 3. Preprocess → JSONL",
    "Builds prompt/messages records with the schema reconstructed from each .sqlite."),
 code("!python -m src.data_prep --source bird --json train/train.json \\",
      "    --db_root train/train_databases --out data/processed/bird_train.jsonl --shuffle",
      "!python -m src.data_prep --source bird --json dev/dev.json \\",
      "    --db_root dev/dev_databases --out data/processed/bird_dev.jsonl"),
 md("## 4. Train (QLoRA)",
    "Main experiment (`exp1`). For a fast sanity run add `--max_steps 30`.",
    "Subsample with `--max_train_samples 5000` if you're time-boxed on Colab."),
 code("!python -m src.train --preset exp1_qwen1.5b_bird_qlora \\",
      "    --train_file data/processed/bird_train.jsonl \\",
      "    --val_file   data/processed/bird_dev.jsonl \\",
      "    --max_train_samples 8000 --epochs 2"),
 md("### (Optional) inline training loop",
    "Equivalent to the CLI above, exposed here so you can tweak interactively."),
 code("from unsloth import FastLanguageModel",
      "from unsloth.chat_templates import train_on_responses_only",
      "from trl import SFTTrainer, SFTConfig",
      "from datasets import load_dataset",
      "",
      "model, tok = FastLanguageModel.from_pretrained(",
      "    'unsloth/Qwen2.5-Coder-1.5B-Instruct', max_seq_length=2048, load_in_4bit=True)",
      "model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,",
      "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],",
      "    use_gradient_checkpointing='unsloth', random_state=3407)",
      "",
      "ds = load_dataset('json', data_files='data/processed/bird_train.jsonl', split='train')",
      "ds = ds.map(lambda e: {'text': [tok.apply_chat_template(m, tokenize=False) for m in e['messages']]},",
      "            batched=True, remove_columns=ds.column_names)",
      "",
      "trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,",
      "    args=SFTConfig(output_dir='outputs/exp1', per_device_train_batch_size=8,",
      "        gradient_accumulation_steps=2, num_train_epochs=2, learning_rate=2e-4,",
      "        lr_scheduler_type='cosine', warmup_ratio=0.03, optim='adamw_8bit',",
      "        logging_steps=20, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),",
      "        dataset_text_field='text', max_seq_length=2048, report_to='none'))",
      "trainer = train_on_responses_only(trainer,",
      "    instruction_part='<|im_start|>user\\n', response_part='<|im_start|>assistant\\n')",
      "stats = trainer.train()",
      "model.save_pretrained('outputs/exp1'); tok.save_pretrained('outputs/exp1')"),
 md("## 5. (Optional) push the adapter to the Hub"),
 code("# from huggingface_hub import login; login()",
      "# model.push_to_hub('<you>/qwen2.5-coder-1.5b-bird-qlora')",
      "# tok.push_to_hub('<you>/qwen2.5-coder-1.5b-bird-qlora')"),
 md("Proceed to **03_inference_eval.ipynb** to measure execution accuracy."),
]

# ===================================================================== #
# 03 — Inference + evaluation
# ===================================================================== #
nb3 = [
 md("# 03 · Inference & Execution-Accuracy Evaluation",
    "",
    "Generate SQL on BIRD dev with the fine-tuned adapter, then score with",
    "execution accuracy (EX), valid-SQL rate and exact match — plus a baseline",
    "(same model, no fine-tuning) for a clean before/after comparison."),
 code("import sys; sys.path.insert(0, '.')"),
 md("## 1. Baseline — base model, zero-shot",
    "Establishes the lift attributable to fine-tuning."),
 code("!python -m src.inference --model_dir unsloth/Qwen2.5-Coder-1.5B-Instruct \\",
      "    --input data/processed/bird_dev.jsonl --output outputs/preds_base.jsonl --limit 200",
      "!python -m src.evaluate --pred outputs/preds_base.jsonl --report outputs/metrics_base.json"),
 md("## 2. Fine-tuned model"),
 code("!python -m src.inference --model_dir outputs/exp1 \\",
      "    --input data/processed/bird_dev.jsonl --output outputs/preds_ft.jsonl --limit 200",
      "!python -m src.evaluate --pred outputs/preds_ft.jsonl --report outputs/metrics_ft.json"),
 md("## 3. Compare"),
 code("import json, pandas as pd",
      "base = json.load(open('outputs/metrics_base.json'))",
      "ft   = json.load(open('outputs/metrics_ft.json'))",
      "tbl = pd.DataFrame({'baseline': base, 'fine-tuned': ft}).loc[",
      "    ['execution_accuracy','valid_sql_rate','exact_match']]",
      "print(tbl)",
      "ax = tbl.plot(kind='bar', rot=0, title='Baseline vs fine-tuned'); ax.figure.tight_layout()",
      "ax.figure.savefig('../report/figures/before_after.png', dpi=120)"),
 md("## 4. Error analysis",
    "Read the captured failures to find systematic mistakes (hallucinated columns,",
    "wrong joins, missing GROUP BY). This is what drives the next experiment."),
 code("for e in ft.get('error_samples', [])[:10]:",
      "    print(e['type'])",
      "    print('  pred:', e.get('pred',''))",
      "    if 'error' in e: print('  err :', e['error'])",
      "    print()"),
 md("## 5. Try the motivating question live"),
 code("from src.inference import load_model, generate_batch",
      "from src.prompts import build_messages, extract_sql",
      "from src.schema_utils import serialize_schema",
      "model, tok, _ = load_model('outputs/exp1', 2048, load_in_4bit=True)",
      "tok.padding_side='left'; tok.pad_token = tok.pad_token or tok.eos_token",
      "schema = serialize_schema('data/sample/db/fintech/fintech.sqlite')",
      "q = 'Who were the top performing merchants last quarter?'",
      "prompt = tok.apply_chat_template(build_messages(schema, q), tokenize=False, add_generation_prompt=True)",
      "print(extract_sql(generate_batch(model, tok, [prompt], 256)[0]))"),
]

write("01_data_exploration.ipynb", nb1)
write("02_finetune_qlora.ipynb", nb2)
write("03_inference_eval.ipynb", nb3)
print("done")
