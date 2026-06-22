"""Preprocess raw Text2SQL datasets into standardized training/eval JSONL.

Each output line is one example:
    {
      "db_id":      str,
      "question":   str,
      "evidence":   str,            # external-knowledge hint (BIRD); "" if none
      "sql":        str,            # gold query
      "difficulty": str,            # simple|moderate|challenging|"" (BIRD dev)
      "db_path":    str,            # path to the .sqlite file (for execution eval)
      "schema":     str,            # serialized schema injected into the prompt
      "messages":   [ {role, content}, ... ]   # system + user + assistant turns
    }

Supported sources
-----------------
* ``bird``   — official BIRD layout (train.json / dev.json + *_databases/).
* ``synsql`` — seeklhy/SynSQL-2.5M style records that already ship DDL strings.
* ``hf``     — any 🤗 dataset with question / sql / schema columns (configurable).

Run ``python -m src.data_prep --help`` for the full CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional

from .prompts import build_messages
from . import schema_utils


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _first_present(d: Dict, keys: List[str], default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _write_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[data_prep] wrote {len(records)} records -> {path}")


def _make_record(
    db_id: str,
    question: str,
    sql: str,
    schema: str,
    db_path: str,
    evidence: str = "",
    difficulty: str = "",
    dialect: str = "SQLite",
) -> Dict:
    return {
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "sql": sql,
        "difficulty": difficulty,
        "db_path": db_path,
        "schema": schema,
        "messages": build_messages(schema, question, evidence, dialect, completion=sql),
    }


# --------------------------------------------------------------------------- #
# BIRD
# --------------------------------------------------------------------------- #
def load_bird(
    json_path: str,
    db_root: Optional[str] = None,
    tables_json: Optional[str] = None,
    dialect: str = "SQLite",
    include_evidence: bool = True,
    schema_mode: str = "ddl",
    limit: Optional[int] = None,
) -> List[Dict]:
    """Parse a BIRD train.json / dev.json file.

    Schema source (pick at least one):
      * ``tables_json`` — BIRD's ``*_tables.json``; schemas built WITHOUT any
        database files. Use this for training to skip the multi-GB ``*_databases``
        download (training never executes SQL, so it doesn't need the .sqlite).
      * ``db_root`` — directory with one folder per db_id holding ``{db_id}.sqlite``
        (BIRD's ``train_databases`` / ``dev_databases``). Required for *execution*
        evaluation, since ``db_path`` must point at a real database.

    When both are given, schemas come from ``tables_json`` and ``db_path`` is set
    from ``db_root`` (so the dev set can still be executed). ``db_path`` is left
    empty when no .sqlite is available — such records are still trainable, just
    not executable.
    """
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if limit:
        raw = raw[:limit]

    # Pre-build schemas from tables.json (db_id -> schema string), if provided.
    tbl_schema: Dict[str, str] = {}
    if tables_json:
        for entry in _read_json_any(tables_json):
            tbl_schema[entry["db_id"]] = schema_utils.schema_from_bird_tables(entry, mode=schema_mode)

    schema_cache: Dict[str, str] = {}
    records: List[Dict] = []
    skipped = 0
    total = len(raw)

    for n, ex in enumerate(raw):
        if n and n % 2000 == 0:
            print(f"[data_prep] processed {n}/{total} examples ...", flush=True)
        db_id = ex["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite") if db_root else ""
        if db_path and not os.path.exists(db_path):
            db_path = ""                      # keep the record, just non-executable

        # Resolve the schema: tables.json first, else read it from the .sqlite.
        if db_id in tbl_schema:
            schema = tbl_schema[db_id]
        elif db_path:
            if db_id not in schema_cache:
                schema_cache[db_id] = schema_utils.serialize_schema(db_path, mode=schema_mode)
            schema = schema_cache[db_id]
        else:
            skipped += 1
            continue

        records.append(_make_record(
            db_id=db_id,
            question=_first_present(ex, ["question"]),
            sql=_first_present(ex, ["SQL", "sql", "query"]),
            schema=schema,
            db_path=db_path,
            evidence=_first_present(ex, ["evidence"]) if include_evidence else "",
            difficulty=_first_present(ex, ["difficulty"]),
            dialect=dialect,
        ))

    if skipped:
        print(f"[data_prep] WARNING: skipped {skipped} examples with no schema source "
              f"(no tables.json entry and no .sqlite). db_root={db_root!r} tables_json={tables_json!r}")
    return records


# --------------------------------------------------------------------------- #
# SynSQL-2.5M  (records carry their own schema as DDL text — no .sqlite needed)
# --------------------------------------------------------------------------- #
def load_synsql(
    json_path: str,
    dialect: str = "SQLite",
    schema_keys: Optional[List[str]] = None,
    question_keys: Optional[List[str]] = None,
    sql_keys: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Parse a SynSQL-style file (.json list or .jsonl).

    Column names vary between mirrors, so the relevant fields are looked up by a
    list of candidate keys (override via the *_keys args if your copy differs).
    """
    schema_keys = schema_keys or ["schema", "ddl", "database_schema", "create_statements", "table_schema"]
    question_keys = question_keys or ["question", "instruction", "nl"]
    sql_keys = sql_keys or ["sql", "SQL", "query", "output"]

    raw = _read_json_any(json_path)
    if limit:
        raw = raw[:limit]

    records: List[Dict] = []
    for ex in raw:
        schema_val = _first_present(ex, schema_keys)
        if isinstance(schema_val, list):
            schema_val = schema_utils.schema_from_create_list(schema_val)
        records.append(_make_record(
            db_id=_first_present(ex, ["db_id", "db", "database"], default="synthetic"),
            question=_first_present(ex, question_keys),
            sql=_first_present(ex, sql_keys),
            schema=str(schema_val),
            db_path="",                       # synthetic: not executable locally
            evidence=_first_present(ex, ["evidence", "external_knowledge"]),
            difficulty=_first_present(ex, ["sql_complexity", "difficulty"]),
            dialect=dialect,
        ))
    return records


def _read_json_any(path: str) -> List[Dict]:
    """Load either a JSON array or a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        head = fh.read(2048)
    is_jsonl = "\n" in head and head.lstrip()[:1] != "["
    with open(path, encoding="utf-8") as fh:
        if is_jsonl:
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Generic 🤗 dataset
# --------------------------------------------------------------------------- #
def load_hf(
    dataset_id: str,
    split: str,
    question_col: str,
    sql_col: str,
    schema_col: str,
    dialect: str = "SQLite",
    limit: Optional[int] = None,
) -> List[Dict]:
    from datasets import load_dataset  # imported lazily; only needed for this source
    ds = load_dataset(dataset_id, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    records = []
    for ex in ds:
        records.append(_make_record(
            db_id=str(ex.get("db_id", "hf")),
            question=str(ex[question_col]),
            sql=str(ex[sql_col]),
            schema=str(ex.get(schema_col, "")),
            db_path="",
            dialect=dialect,
        ))
    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess Text2SQL data into JSONL.")
    p.add_argument("--source", choices=["bird", "synsql", "hf"], required=True)
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--dialect", default="SQLite")
    p.add_argument("--schema_mode", default="ddl", choices=["ddl", "compact"])
    p.add_argument("--no_evidence", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=3407)

    # bird
    p.add_argument("--json", help="path to train.json / dev.json (bird, synsql)")
    p.add_argument("--db_root", help="dir with one folder per db_id (bird; needed for execution eval)")
    p.add_argument("--tables_json", help="BIRD *_tables.json; build schemas without databases (training)")

    # hf
    p.add_argument("--hf_dataset")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--question_col", default="question")
    p.add_argument("--sql_col", default="sql")
    p.add_argument("--schema_col", default="schema")

    args = p.parse_args()

    if args.source == "bird":
        assert args.json, "--json is required for bird"
        assert args.db_root or args.tables_json, (
            "provide --tables_json (schemas without databases; for training) "
            "and/or --db_root (real .sqlite; needed for execution eval)")
        records = load_bird(
            args.json, db_root=args.db_root, tables_json=args.tables_json,
            dialect=args.dialect, include_evidence=not args.no_evidence,
            schema_mode=args.schema_mode, limit=args.limit,
        )
    elif args.source == "synsql":
        assert args.json, "--json is required for synsql"
        records = load_synsql(args.json, dialect=args.dialect, limit=args.limit)
    else:
        assert args.hf_dataset, "--hf_dataset is required for hf"
        records = load_hf(
            args.hf_dataset, args.hf_split, args.question_col,
            args.sql_col, args.schema_col, dialect=args.dialect, limit=args.limit,
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(records)

    _write_jsonl(records, args.out)


if __name__ == "__main__":
    main()
