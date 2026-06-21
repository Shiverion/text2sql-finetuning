"""Evaluate Text2SQL predictions against SQLite databases.

Metrics (BIRD-style):
  * execution_accuracy (EX) — predicted query runs AND returns the same result
    set as the gold query (order-insensitive, the standard EX definition).
  * valid_sql_rate          — predicted query executes without raising an error.
  * exact_match             — normalized string equality with the gold query
    (a strict lower bound; many correct queries differ textually).

Results are also broken down by BIRD difficulty when available.

Usage:
    python -m src.evaluate --pred outputs/preds_bird_dev.jsonl \
        --report outputs/metrics_bird_dev.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# SQL execution with a hard timeout (runaway / cartesian-product queries)
# --------------------------------------------------------------------------- #
class ExecResult:
    __slots__ = ("ok", "rows", "error")

    def __init__(self, ok: bool, rows=None, error: str = ""):
        self.ok = ok
        self.rows = rows
        self.error = error


def execute_sql(db_path: str, sql: str, timeout_s: float = 30.0) -> ExecResult:
    """Run ``sql`` read-only against the sqlite file with a wall-clock timeout."""
    if not sql or not sql.strip():
        return ExecResult(False, error="empty query")
    holder: Dict[str, ExecResult] = {}

    def _run():
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            try:
                cur = conn.execute(sql)
                holder["res"] = ExecResult(True, rows=cur.fetchall())
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001 - want every failure mode as "invalid"
            holder["res"] = ExecResult(False, error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return ExecResult(False, error=f"timeout>{timeout_s}s")
    return holder.get("res", ExecResult(False, error="unknown"))


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _as_multiset(rows: List[Tuple]) -> Counter:
    """Order-insensitive comparison. Cast every cell to str so 1 == '1'
    mismatches in column typing don't cause spurious failures."""
    return Counter(tuple(str(c) for c in row) for row in rows)


def result_sets_match(gold_rows, pred_rows) -> bool:
    return _as_multiset(gold_rows) == _as_multiset(pred_rows)


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower().strip()


def exact_match(gold: str, pred: str) -> bool:
    return normalize_sql(gold) == normalize_sql(pred)


# --------------------------------------------------------------------------- #
# Scoring a prediction file
# --------------------------------------------------------------------------- #
def score(preds: List[Dict], timeout_s: float = 30.0) -> Dict:
    n = len(preds)
    ex_hits = valid_hits = em_hits = 0
    by_diff = defaultdict(lambda: {"n": 0, "ex": 0})
    errors: List[Dict] = []

    for i, p in enumerate(preds):
        gold_sql, pred_sql = p.get("gold_sql", ""), p.get("pred_sql", "")
        db_path = p.get("db_path", "")
        diff = p.get("difficulty", "") or "unknown"
        by_diff[diff]["n"] += 1

        if exact_match(gold_sql, pred_sql):
            em_hits += 1

        # Without a database we can only score exact match (e.g. SynSQL).
        if not db_path:
            continue

        pred_res = execute_sql(db_path, pred_sql, timeout_s)
        if pred_res.ok:
            valid_hits += 1
            gold_res = execute_sql(db_path, gold_sql, timeout_s)
            if gold_res.ok and result_sets_match(gold_res.rows, pred_res.rows):
                ex_hits += 1
                by_diff[diff]["ex"] += 1
            elif len(errors) < 50:
                errors.append({"i": i, "type": "wrong_result",
                               "gold": gold_sql, "pred": pred_sql})
        elif len(errors) < 50:
            errors.append({"i": i, "type": "exec_error",
                           "error": pred_res.error, "pred": pred_sql})

    metrics = {
        "n": n,
        "execution_accuracy": round(ex_hits / n, 4) if n else 0.0,
        "valid_sql_rate": round(valid_hits / n, 4) if n else 0.0,
        "exact_match": round(em_hits / n, 4) if n else 0.0,
        "counts": {"ex": ex_hits, "valid": valid_hits, "exact": em_hits},
        "by_difficulty": {
            d: {"n": v["n"], "execution_accuracy": round(v["ex"] / v["n"], 4) if v["n"] else 0.0}
            for d, v in sorted(by_diff.items())
        },
        "error_samples": errors[:50],
    }
    return metrics


def print_metrics(m: Dict) -> None:
    print("=" * 52)
    print(f"  examples           : {m['n']}")
    print(f"  execution accuracy : {m['execution_accuracy']:.2%}  ({m['counts']['ex']})")
    print(f"  valid SQL rate     : {m['valid_sql_rate']:.2%}  ({m['counts']['valid']})")
    print(f"  exact match        : {m['exact_match']:.2%}  ({m['counts']['exact']})")
    if m["by_difficulty"]:
        print("  by difficulty:")
        for d, v in m["by_difficulty"].items():
            print(f"     {d:<12} n={v['n']:<6} EX={v['execution_accuracy']:.2%}")
    print("=" * 52)


def main() -> None:
    p = argparse.ArgumentParser(description="Score Text2SQL predictions.")
    p.add_argument("--pred", required=True, help="predictions JSONL")
    p.add_argument("--report", default=None, help="write metrics JSON here")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    preds = []
    with open(args.pred, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                preds.append(json.loads(line))

    m = score(preds, timeout_s=args.timeout)
    print_metrics(m)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)
        print(f"[evaluate] wrote {args.report}")


if __name__ == "__main__":
    main()
