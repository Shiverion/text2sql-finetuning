"""Create a tiny self-contained fintech SQLite database + BIRD-format examples.

This exists so the *entire* pipeline (preprocess -> prompt -> execution-eval) can
be exercised with zero downloads and zero GPU — useful for CI, for verifying the
code on any laptop, and as a minimal worked example of the data format the model
is trained on. The schema is deliberately fintech-flavoured to match the brief
("top performing merchants in the last quarter").

Output layout (mirrors BIRD's `*_databases/<db_id>/<db_id>.sqlite`):
    data/sample/db/fintech/fintech.sqlite
    data/sample/examples.json          # list of {db_id, question, evidence, SQL, difficulty}
"""
from __future__ import annotations

import json
import os
import sqlite3

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
DB_DIR = os.path.join(ROOT, "db", "fintech")
DB_PATH = os.path.join(DB_DIR, "fintech.sqlite")
EXAMPLES_PATH = os.path.join(ROOT, "examples.json")


SCHEMA_SQL = """
CREATE TABLE merchants (
    merchant_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT,
    city          TEXT,
    onboarded_at  TEXT
);
CREATE TABLE transactions (
    txn_id        INTEGER PRIMARY KEY,
    merchant_id   INTEGER NOT NULL,
    amount        REAL NOT NULL,
    status        TEXT,          -- 'settled' | 'refunded' | 'failed'
    txn_date      TEXT,          -- ISO date 'YYYY-MM-DD'
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);
"""

MERCHANTS = [
    (1, "Kopi Kenangan", "F&B", "Jakarta", "2023-01-10"),
    (2, "Tokopedia Mart", "Retail", "Jakarta", "2022-11-02"),
    (3, "GoFood Partner",  "F&B", "Bandung", "2023-03-15"),
    (4, "Bluebird Pay",    "Transport", "Surabaya", "2021-07-21"),
    (5, "Sayurbox",        "Retail", "Bandung", "2023-06-30"),
]

# (txn_id, merchant_id, amount, status, txn_date)
TRANSACTIONS = [
    (1, 1, 250000, "settled",  "2024-01-05"),
    (2, 1, 180000, "settled",  "2024-02-11"),
    (3, 1,  90000, "refunded", "2024-03-02"),
    (4, 2, 500000, "settled",  "2024-01-20"),
    (5, 2, 750000, "settled",  "2024-03-28"),
    (6, 3, 120000, "settled",  "2024-02-14"),
    (7, 3, 130000, "failed",   "2024-02-15"),
    (8, 4, 60000,  "settled",  "2024-03-09"),
    (9, 5, 300000, "settled",  "2024-03-31"),
    (10, 5, 220000, "settled", "2023-12-15"),
    (11, 2, 410000, "settled", "2024-03-30"),
    (12, 1, 175000, "settled", "2024-03-18"),
]

EXAMPLES = [
    {
        "db_id": "fintech",
        "question": "Who were the top 3 performing merchants by total settled "
                    "transaction amount in Q1 2024?",
        "evidence": "Q1 2024 means txn_date between 2024-01-01 and 2024-03-31; "
                    "performance is the sum of amount where status = 'settled'.",
        "SQL": "SELECT m.name, SUM(t.amount) AS total "
               "FROM merchants m JOIN transactions t ON m.merchant_id = t.merchant_id "
               "WHERE t.status = 'settled' AND t.txn_date BETWEEN '2024-01-01' AND '2024-03-31' "
               "GROUP BY m.merchant_id ORDER BY total DESC LIMIT 3",
        "difficulty": "moderate",
    },
    {
        "db_id": "fintech",
        "question": "How many merchants are in the Retail category?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM merchants WHERE category = 'Retail'",
        "difficulty": "simple",
    },
    {
        "db_id": "fintech",
        "question": "What is the total refunded amount across all merchants?",
        "evidence": "Refunded transactions have status = 'refunded'.",
        "SQL": "SELECT SUM(amount) FROM transactions WHERE status = 'refunded'",
        "difficulty": "simple",
    },
    {
        "db_id": "fintech",
        "question": "List each city and its number of merchants, most merchants first.",
        "evidence": "",
        "SQL": "SELECT city, COUNT(*) AS n FROM merchants GROUP BY city ORDER BY n DESC",
        "difficulty": "simple",
    },
    {
        "db_id": "fintech",
        "question": "Which merchant had the highest single settled transaction, and how much?",
        "evidence": "",
        "SQL": "SELECT m.name, t.amount FROM merchants m "
               "JOIN transactions t ON m.merchant_id = t.merchant_id "
               "WHERE t.status = 'settled' ORDER BY t.amount DESC LIMIT 1",
        "difficulty": "moderate",
    },
]


def build() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    conn.executemany("INSERT INTO merchants VALUES (?,?,?,?,?)", MERCHANTS)
    conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?)", TRANSACTIONS)
    conn.commit()
    conn.close()

    with open(EXAMPLES_PATH, "w", encoding="utf-8") as fh:
        json.dump(EXAMPLES, fh, indent=2)

    print(f"[make_sample_data] db        -> {os.path.normpath(DB_PATH)}")
    print(f"[make_sample_data] examples  -> {os.path.normpath(EXAMPLES_PATH)} "
          f"({len(EXAMPLES)} questions)")


if __name__ == "__main__":
    build()
