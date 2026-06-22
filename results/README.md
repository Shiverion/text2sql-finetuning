# Results — Experiment 1 (executed run)

Fine-tuned **Qwen2.5-Coder-1.5B + BIRD (QLoRA)** on a free **Kaggle T4**, evaluated
on **200 BIRD-dev** questions with execution accuracy. Adapter on the Hub:
[`Shiverion/qwen2.5-coder-1.5b-bird-qlora`](https://huggingface.co/Shiverion/qwen2.5-coder-1.5b-bird-qlora).

| Run | Execution acc. (EX) | Valid-SQL rate | Exact match |
|---|---|---|---|
| Baseline (1.5B, zero-shot) | 14.0% (28/200) | 40.0% (80/200) | 0.0% (0/200) |
| **Fine-tuned (QLoRA)** | **15.5% (31/200)** | **73.5% (147/200)** | **2.0% (4/200)** |

EX by difficulty (fine-tuned): simple **21.9%**, moderate **7.4%**, challenging **0%**.

**Headline:** fine-tuning nearly doubled the **valid-SQL rate (40% → 73.5%)** — the
model learned to emit clean, executable, schema-grounded SQL (the brief's priority).
EX gains are modest, as expected for a small/quick run; see
[`../report/REPORT.md`](../report/REPORT.md) §8–9 for full analysis and the chart
[`../report/figures/before_after.png`](../report/figures/before_after.png).

- `03_inference_eval.executed.ipynb` — the executed evaluation notebook (with
  outputs) from the Kaggle run. The clean, re-runnable source is
  [`../notebooks/03_inference_eval.ipynb`](../notebooks/03_inference_eval.ipynb).
