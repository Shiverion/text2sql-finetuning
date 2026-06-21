"""Prompt construction for Text2SQL.

Kept dependency-free (no transformers import) so it can be reused by data prep,
training, inference and tests. The actual chat-template rendering happens in the
train / inference scripts where a tokenizer is available; here we only produce
the role/content message list, which is the single source of truth for prompt
formatting across training and inference (consistency matters — a train/serve
prompt mismatch is the most common silent accuracy killer).
"""
from __future__ import annotations

from typing import List, Dict, Optional


SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct, executable SQL. "
    "Given a database schema and a question, output a single {dialect} SQL "
    "query that answers the question. Use only the tables and columns in the "
    "schema. Return only the SQL query with no explanation."
)


USER_TEMPLATE = """Database schema:
{schema}

{evidence_block}Question: {question}

SQL:"""


def build_user_message(
    schema: str,
    question: str,
    evidence: Optional[str] = None,
    dialect: str = "SQLite",
) -> str:
    evidence_block = ""
    if evidence and evidence.strip():
        # BIRD's "evidence" is external knowledge (e.g. a formula or mapping)
        # that the question relies on — feeding it in is worth several points.
        evidence_block = f"Hint: {evidence.strip()}\n\n"
    return USER_TEMPLATE.format(
        schema=schema.strip(),
        evidence_block=evidence_block,
        question=question.strip(),
    )


def build_messages(
    schema: str,
    question: str,
    evidence: Optional[str] = None,
    dialect: str = "SQLite",
    completion: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return chat messages. If ``completion`` is given, append the assistant
    turn (training); otherwise stop after the user turn (inference)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(dialect=dialect)},
        {"role": "user", "content": build_user_message(schema, question, evidence, dialect)},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": format_sql_completion(completion)})
    return messages


def format_sql_completion(sql: str) -> str:
    """Normalize the gold SQL into the exact string we train the model to emit."""
    sql = sql.strip().rstrip(";").strip()
    return f"```sql\n{sql}\n```"


# --------------------------------------------------------------------------- #
# Output parsing (inverse of format_sql_completion)
# --------------------------------------------------------------------------- #
import re

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model generation.

    Handles fenced ```sql blocks, a leading 'SQL:' label, and bare output.
    Returns the first statement, trailing semicolon stripped.
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text

    # Drop a leading 'SQL:' label if the model echoed it.
    candidate = re.sub(r"^\s*SQL\s*:\s*", "", candidate, flags=re.IGNORECASE)

    # Keep only up to the first statement terminator to avoid trailing chatter.
    candidate = candidate.strip()
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0]
    # If the model kept generating prose after the query, cut at a blank line.
    candidate = candidate.split("\n\n", 1)[0]
    return candidate.strip()
