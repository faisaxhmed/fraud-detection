"""SQLite audit trail for /review agent decisions - every decision the agent loop makes is
logged here: what the transaction looked like, which tools it called and what they returned,
and the final verdict. This is the compliance record, not a debugging log, so it's append-only
and never modifies or deletes prior rows."""
import json
import os
import sqlite3
from datetime import datetime, timezone

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BACKEND_DIR, "audit.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            transaction_summary TEXT NOT NULL,
            shap_top_factors TEXT NOT NULL,
            tools_called TEXT NOT NULL,
            decision TEXT NOT NULL,
            reasoning TEXT NOT NULL
        )
        """
    )
    return conn


def log_decision(
    transaction_summary: dict,
    shap_top_factors: list,
    tools_called: list,
    decision: str,
    reasoning: str,
) -> None:
    """Appends one row to the audit log. Called once per /review request, after the agent
    loop has reached a final decision (including the 5-iteration-cap fallback)."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO review_audit_log
                (timestamp, transaction_summary, shap_top_factors, tools_called, decision, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(transaction_summary),
                json.dumps(shap_top_factors),
                json.dumps(tools_called),
                decision,
                reasoning,
            ),
        )
        conn.commit()
    finally:
        conn.close()
