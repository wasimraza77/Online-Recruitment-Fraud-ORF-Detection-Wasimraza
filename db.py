"""SQLite database module for user authentication.

Manages a lightweight SQLite database for user registration and login.
The database file is stored at FraudRecruitment/users.db and is auto-created
on first run with a seed admin account.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "users.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT  NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SEED_ADMIN = {
    "full_name": "Admin",
    "email": "admin@fraud.local",
    "password": "Admin123",
}


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the users table if it doesn't exist and seed the admin account."""
    conn = _get_conn()
    try:
        conn.executescript(_SCHEMA)
        # Seed admin if the table is empty
        row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
        if row["cnt"] == 0:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (
                    SEED_ADMIN["full_name"],
                    SEED_ADMIN["email"],
                    generate_password_hash(SEED_ADMIN["password"]),
                    datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
                ),
            )
            conn.commit()
            logger.info("Seeded admin account: %s", SEED_ADMIN["email"])
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

def validate_password(password: str) -> list[str]:
    """Return a list of validation error messages. Empty list = valid."""
    errors: list[str] = []
    if len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter.")
    if not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one number.")
    return errors


def validate_email(email: str) -> bool:
    """Basic email format check."""
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email))


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def create_user(full_name: str, email: str, password: str) -> dict | None:
    """Create a new user and return their data dict, or None if email already exists."""
    conn = _get_conn()
    try:
        now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
        conn.execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name.strip(), email.strip().lower(), generate_password_hash(password), now),
        )
        conn.commit()
        return get_user_by_email(email)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    """Look up a user by email. Returns a plain dict or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, full_name, email, created_at FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def verify_login(email: str, password: str) -> dict | None:
    """Check email + password against the database. Returns user dict or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, full_name, email, password_hash, created_at FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            return {"id": row["id"], "full_name": row["full_name"], "email": row["email"], "created_at": row["created_at"]}
        return None
    finally:
        conn.close()


def get_user_count() -> int:
    """Return total number of registered users."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
        return row["cnt"]
    finally:
        conn.close()
