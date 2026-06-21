"""Read and serialize SQLite database schemas for use in prompts.

The model can only generate a correct query if it is told what tables and
columns exist. We reconstruct the schema directly from the ``.sqlite`` file so
it always matches the database the query is later executed against.

Two serialization modes:
  * ``ddl``     -> the original ``CREATE TABLE`` statements (most informative).
  * ``compact`` -> ``table(col type, col type, ...)`` one line per table (cheap
                   on tokens, useful for very wide schemas or small context).
"""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    # read-only URI connection so we never mutate the eval databases.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def list_tables(db_path: str) -> List[str]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_create_statements(db_path: str) -> Dict[str, str]:
    """Return {table_name: CREATE TABLE statement} for all user tables."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
        return {name: sql.strip() for name, sql in rows}
    finally:
        conn.close()


def _column_info(conn: sqlite3.Connection, table: str) -> List[tuple]:
    # PRAGMA table_info -> (cid, name, type, notnull, dflt_value, pk)
    return conn.execute(f'PRAGMA table_info("{table}")').fetchall()


def get_compact_schema(db_path: str) -> str:
    """`table(col TYPE, col TYPE PK, ...)` — one line per table."""
    conn = _connect(db_path)
    lines: List[str] = []
    try:
        for table in list_tables(db_path):
            cols = []
            for _, name, ctype, _notnull, _dflt, pk in _column_info(conn, table):
                tag = f"{name} {ctype or 'TEXT'}".strip()
                if pk:
                    tag += " PK"
                cols.append(tag)
            lines.append(f"{table}({', '.join(cols)})")
    finally:
        conn.close()
    return "\n".join(lines)


def sample_rows(db_path: str, table: str, n: int = 3) -> List[tuple]:
    """A few example rows — optional context that helps with value formatting."""
    conn = _connect(db_path)
    try:
        return conn.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}').fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def serialize_schema(
    db_path: str,
    mode: str = "ddl",
    with_samples: bool = False,
    sample_n: int = 3,
) -> str:
    """Top-level entry point used by data_prep / inference.

    Parameters
    ----------
    mode : "ddl" | "compact"
    with_samples : append a few example rows per table (DDL mode only).
    """
    if mode == "compact":
        return get_compact_schema(db_path)

    creates = get_create_statements(db_path)
    blocks: List[str] = []
    for table, ddl in creates.items():
        block = _normalize_ddl(ddl)
        if with_samples:
            rows = sample_rows(db_path, table, sample_n)
            if rows:
                preview = "; ".join(str(r) for r in rows)
                block += f"\n/* {sample_n} example rows: {preview} */"
        blocks.append(block)
    return "\n\n".join(blocks)


def _normalize_ddl(ddl: str) -> str:
    """Collapse noisy whitespace inside a CREATE statement for compactness."""
    ddl = re.sub(r"[ \t]+", " ", ddl)
    ddl = re.sub(r"\n\s*\n+", "\n", ddl)
    if not ddl.rstrip().endswith(";"):
        ddl = ddl.rstrip() + ";"
    return ddl.strip()


def schema_from_create_list(create_statements: List[str]) -> str:
    """For datasets (e.g. SynSQL) that already ship DDL strings instead of a DB."""
    return "\n\n".join(_normalize_ddl(s) for s in create_statements if s and s.strip())


if __name__ == "__main__":  # tiny manual check
    import sys
    if len(sys.argv) > 1:
        print(serialize_schema(sys.argv[1], mode="ddl"))
