"""Generate the three deliverable notebooks as valid .ipynb JSON.

Run once: `python scripts/_gen_notebooks.py`. Kept in the repo so the notebooks
are reproducible/diff-able rather than hand-edited binary-ish JSON.

All three notebooks open with the same idempotent SETUP cell: on Colab it clones
the repo and `chdir`s into it; locally (run from the repo root) it is a no-op.
After SETUP, every path is repo-root-relative and `import src...` works.
"""
import json, os

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

REPO_URL = "https://github.com/Shiverion/text2sql-finetuning.git"
REPO_DIR = "text2sql-finetuning"

# Shared, environment-agnostic setup. Uses os.chdir (not the %cd magic) so it is
# valid inside the if-block and changes the *kernel* working directory on Colab.
SETUP = [
    "# --- SETUP: run me first (works on Colab AND locally) ---",
    "import os, sys, subprocess",
    "if not os.path.exists('src') and os.path.basename(os.getcwd()) != '{d}':".format(d=REPO_DIR),
    "    if not os.path.isdir('{d}'):".format(d=REPO_DIR),
    "        subprocess.run(['git', 'clone', '{url}'], check=True)".format(url=REPO_URL),
    "    os.chdir('{d}')".format(d=REPO_DIR),
    "sys.path.insert(0, os.getcwd())",
    "print('working dir :', os.getcwd())",
    "print('src present :', os.path.isdir('src'))",
]


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
 code(*SETUP),
 code("!pip install -q datasets pandas matplotlib"),
 code("import os, json, glob, re",
      "import pandas as pd",
      "import matplotlib.pyplot as plt",
      "FIG = 'report/figures'; os.makedirs(FIG, exist_ok=True)"),
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
      "    print('BIRD not found - using bundled sample')",
      "    return json.load(open('data/sample/examples.json', encoding='utf-8')), 'sample'",
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
 code("# Column counts per DB. Sample DB by default; on full BIRD point the glob",
      "# at dev_databases/**/*.sqlite.",
      "import sqlite3",
      "dbs = glob.glob('data/sample/db/**/*.sqlite', recursive=True) or glob.glob('dev_databases/**/*.sqlite', recursive=True)",
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
    "> **First:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all."),
 code("import torch; print('CUDA:', torch.cuda.is_available(),",
      "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(enable the T4 GPU runtime!)')"),
 md("## 1. Get the code (clone the repo)"),
 code(*SETUP),
 md("## 2. Install Unsloth + training stack",
    "`pip install unsloth` pulls a torch/transformers/trl/peft combo that's tested",
    "together — don't hand-pin these on Colab unless you know why. (~2–4 min.)"),
 code("!pip install -q unsloth",
      "# If you hit a version error later, get the matching nightly instead:",
      "# !pip install -q --upgrade --no-deps \"unsloth @ git+https://github.com/unslothai/unsloth.git\" \\",
      "#     \"unsloth_zoo @ git+https://github.com/unslothai/unsloth_zoo.git\""),
 md("## 3. Download BIRD",
    "If you'd rather skip this, upload your own processed JSONL and jump to step 5."),
 code("!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip && unzip -q -o train.zip",
      "!wget -q https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip   && unzip -q -o dev.zip"),
 md("### Check the extracted layout",
    "BIRD's archive layout shifts between releases. Look at the tree below and set",
    "`TRAIN_JSON / TRAIN_DB / DEV_JSON / DEV_DB` to match what you actually see",
    "(you want the folder that contains one sub-folder per database, each holding",
    "`<db_id>.sqlite`)."),
 code("import glob, os",
      "for p in sorted(glob.glob('train/*'))[:10] + sorted(glob.glob('dev*/*'))[:10]:",
      "    print(p)",
      "print('--- sample sqlite files ---')",
      "for p in glob.glob('**/*.sqlite', recursive=True)[:5]:",
      "    print(p)"),
 code("# Edit these to match the tree printed above:",
      "TRAIN_JSON = 'train/train.json'",
      "TRAIN_DB   = 'train/train_databases'",
      "DEV_JSON   = 'dev/dev.json'",
      "DEV_DB     = 'dev/dev_databases'"),
 md("## 4. Preprocess → JSONL",
    "Builds prompt/messages records with the schema reconstructed from each .sqlite."),
 code("!python -m src.data_prep --source bird --json {TRAIN_JSON} --db_root {TRAIN_DB} \\",
      "    --out data/processed/bird_train.jsonl --shuffle",
      "!python -m src.data_prep --source bird --json {DEV_JSON} --db_root {DEV_DB} \\",
      "    --out data/processed/bird_dev.jsonl"),
 code("# peek at one processed record",
      "import json",
      "print(json.dumps(json.loads(open('data/processed/bird_train.jsonl').readline()), indent=2)[:1200])"),
 md("## 5. Train (QLoRA)",
    "Runs the hardened trainer in `src/train.py` (Unsloth + completion-only loss;",
    "the SFTConfig/SFTTrainer kwargs auto-adapt to the installed TRL version).",
    "`--max_train_samples` keeps a free-tier run inside the session limit; drop it",
    "for the full set. For a 2-minute sanity check first, add `--max_steps 30`."),
 code("!python -m src.train --preset exp1_qwen1.5b_bird_qlora \\",
      "    --train_file data/processed/bird_train.jsonl \\",
      "    --val_file   data/processed/bird_dev.jsonl \\",
      "    --max_train_samples 8000 --epochs 2"),
 md("The adapter + tokenizer are saved to `outputs/qwen2.5-coder-1.5b-bird-qlora/`.",
    "The training implementation lives in [`src/train.py`](src/train.py); tweak",
    "hyper-parameters via the presets in [`src/config.py`](src/config.py)."),
 md("## 6. (Optional) push the adapter to the Hub"),
 code("# from huggingface_hub import login; login()",
      "# from unsloth import FastLanguageModel  # already imported during training",
      "# model.push_to_hub_merged(...)  # or just upload the outputs/ folder",
      "# !huggingface-cli upload <you>/qwen2.5-coder-1.5b-bird-qlora outputs/qwen2.5-coder-1.5b-bird-qlora"),
 md("Now run **03_inference_eval.ipynb** to measure execution accuracy."),
]

# ===================================================================== #
# 03 — Inference + evaluation
# ===================================================================== #
nb3 = [
 md("# 03 · Inference & Execution-Accuracy Evaluation",
    "",
    "Generate SQL on BIRD dev with the fine-tuned adapter, then score with",
    "execution accuracy (EX), valid-SQL rate and exact match — plus a baseline",
    "(same model, no fine-tuning) for a clean before/after comparison.",
    "",
    "> Run this in the **same Colab session** as notebook 02 (so `outputs/` and",
    "> `data/processed/` exist), or re-run notebook 02's SETUP + preprocessing first."),
 code(*SETUP),
 md("## 1. Baseline — base model, zero-shot",
    "Establishes the lift attributable to fine-tuning. (`--limit 200` keeps it",
    "quick; remove for the full dev set.)"),
 code("!python -m src.inference --model_dir unsloth/Qwen2.5-Coder-1.5B-Instruct \\",
      "    --input data/processed/bird_dev.jsonl --output outputs/preds_base.jsonl --limit 200",
      "!python -m src.evaluate --pred outputs/preds_base.jsonl --report outputs/metrics_base.json"),
 md("## 2. Fine-tuned model"),
 code("!python -m src.inference --model_dir outputs/qwen2.5-coder-1.5b-bird-qlora \\",
      "    --input data/processed/bird_dev.jsonl --output outputs/preds_ft.jsonl --limit 200",
      "!python -m src.evaluate --pred outputs/preds_ft.jsonl --report outputs/metrics_ft.json"),
 md("## 3. Compare"),
 code("import json, pandas as pd, os",
      "os.makedirs('report/figures', exist_ok=True)",
      "base = json.load(open('outputs/metrics_base.json'))",
      "ft   = json.load(open('outputs/metrics_ft.json'))",
      "tbl = pd.DataFrame({'baseline': base, 'fine-tuned': ft}).loc[",
      "    ['execution_accuracy','valid_sql_rate','exact_match']]",
      "print(tbl)",
      "ax = tbl.plot(kind='bar', rot=0, title='Baseline vs fine-tuned'); ax.figure.tight_layout()",
      "ax.figure.savefig('report/figures/before_after.png', dpi=120)"),
 md("## 4. Error analysis",
    "Read the captured failures to find systematic mistakes (hallucinated columns,",
    "wrong joins, missing GROUP BY). This is what drives the next experiment."),
 code("for e in ft.get('error_samples', [])[:10]:",
      "    print(e['type'])",
      "    print('  pred:', e.get('pred',''))",
      "    if 'error' in e: print('  err :', e['error'])",
      "    print()"),
 md("## 5. Try the motivating question live",
    "Uses the bundled fintech DB, so it works even without BIRD downloaded."),
 code("from src.inference import load_model, generate_batch",
      "from src.prompts import build_messages, extract_sql",
      "from src.schema_utils import serialize_schema",
      "model, tok, _ = load_model('outputs/qwen2.5-coder-1.5b-bird-qlora', 2048, load_in_4bit=True)",
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
