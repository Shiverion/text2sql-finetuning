"""End-to-end smoke test of the non-GPU pipeline (no ML deps, no downloads).

Validates the parts that don't need a GPU and that are easy to get subtly wrong:
  1. sample DB + examples build,
  2. data_prep turns them into well-formed prompt/messages records,
  3. the execution-evaluation logic scores correctly:
       - feeding the GOLD queries back as "predictions" must give 100% EX,
       - feeding a deliberately BROKEN query must drop EX and valid-SQL.

Run:  python scripts/smoke_test.py
Exits non-zero on any failure so it can double as a CI gate.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.make_sample_data import build, DB_PATH, EXAMPLES_PATH  # noqa: E402
from src.data_prep import load_bird                                  # noqa: E402
from src.evaluate import score                                       # noqa: E402
from src.prompts import extract_sql                                  # noqa: E402


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise SystemExit(f"smoke test failed: {msg}")


def main() -> None:
    print("1) build sample data")
    build()
    db_root = os.path.join(os.path.dirname(EXAMPLES_PATH), "db")

    print("\n2) preprocess (data_prep.load_bird)")
    records = load_bird(EXAMPLES_PATH, db_root, schema_mode="ddl")
    check(len(records) == 5, f"loaded {len(records)} records (expected 5)")
    r0 = records[0]
    check("merchants" in r0["schema"] and "transactions" in r0["schema"],
          "schema contains both tables")
    check(r0["messages"][0]["role"] == "system", "first message is system")
    check(r0["messages"][-1]["role"] == "assistant", "last message is assistant (gold SQL)")
    check("```sql" in r0["messages"][-1]["content"], "gold completion is fenced SQL")

    print("\n3) extract_sql round-trips the fenced completion")
    recovered = extract_sql(r0["messages"][-1]["content"])
    check(recovered.lower().startswith("select"), "extract_sql recovers a SELECT")

    print("\n4) evaluate GOLD-as-prediction (should be perfect)")
    perfect = [{
        "db_id": r["db_id"], "gold_sql": r["sql"], "pred_sql": r["sql"],
        "db_path": r["db_path"], "difficulty": r["difficulty"],
    } for r in records]
    m_perfect = score(perfect)
    check(m_perfect["execution_accuracy"] == 1.0, "EX == 100% when pred == gold")
    check(m_perfect["valid_sql_rate"] == 1.0, "valid SQL == 100% when pred == gold")

    print("\n5) evaluate a BROKEN prediction (should be caught)")
    broken = [dict(p) for p in perfect]
    broken[0]["pred_sql"] = "SELECT * FROM nonexistent_table"       # exec error
    broken[1]["pred_sql"] = "SELECT 999"                             # valid but wrong result
    m_broken = score(broken)
    check(m_broken["execution_accuracy"] < 1.0, "EX drops when predictions are wrong")
    check(m_broken["valid_sql_rate"] < 1.0, "valid-SQL drops on an exec error")
    check(m_broken["counts"]["ex"] == 3, "exactly the 3 correct predictions count as EX")

    print("\n6) the motivating example resolves to a sensible top-merchant query")
    print("    Q:", records[0]["question"])
    print("    A:", records[0]["sql"])

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
