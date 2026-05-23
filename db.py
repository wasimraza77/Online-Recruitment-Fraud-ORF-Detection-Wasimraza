"""Database module for user authentication supporting both SQLite and PostgreSQL.

Manages user registration and login.
If the environment variable DATABASE_URL is defined (e.g. on Neon/Supabase),
it uses PostgreSQL. Otherwise, it falls back to a local SQLite database file at
users.db.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "users.db"

# Check if we should use PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_POSTGRES = bool(DATABASE_URL)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT  NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    full_name     VARCHAR(255) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at    VARCHAR(255) NOT NULL
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

def _get_conn():
    if IS_POSTGRES:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

def _execute(query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False):
    conn = _get_conn()
    try:
        if IS_POSTGRES:
            import psycopg2.extras
            # Convert SQLite placeholders '?' to '%s'
            query = query.replace("?", "%s")
            
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query, params)
            
            result = None
            if fetch_one:
                result = cur.fetchone()
                if result:
                    result = dict(result)
            elif fetch_all:
                result = [dict(r) for r in cur.fetchall()]
                
            if commit:
                conn.commit()
            return result
        else:
            # SQLite
            cur = conn.cursor()
            cur.execute(query, params)
            
            result = None
            if fetch_one:
                row = cur.fetchone()
                if row:
                    result = dict(row)
            elif fetch_all:
                result = [dict(r) for r in cur.fetchall()]
                
            if commit:
                conn.commit()
            return result
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the users table if it doesn't exist and seed the admin account."""
    conn = _get_conn()
    try:
        if IS_POSTGRES:
            cur = conn.cursor()
            cur.execute(_POSTGRES_SCHEMA)
            conn.commit()
            logger.info("PostgreSQL database connection initialized successfully.")
        else:
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
            logger.info("Local SQLite database initialized successfully.")
            
        # Seed admin if the table is empty
        row = _execute("SELECT COUNT(*) AS cnt FROM users", fetch_one=True)
        if row and row["cnt"] == 0:
            now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
            _execute(
                "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (
                    SEED_ADMIN["full_name"],
                    SEED_ADMIN["email"],
                    generate_password_hash(SEED_ADMIN["password"]),
                    now,
                ),
                commit=True
            )
            logger.info("Seeded default admin account: %s", SEED_ADMIN["email"])
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
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    try:
        _execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name.strip(), email.strip().lower(), generate_password_hash(password), now),
            commit=True
        )
        return get_user_by_email(email)
    except Exception as exc:
        # Gracefully catch unique constraint/integrity errors from both backends
        err_msg = str(exc).lower()
        if "unique" in err_msg or "duplicate" in err_msg or "integrity" in err_msg:
            return None
        raise exc

def get_user_by_email(email: str) -> dict | None:
    """Look up a user by email. Returns a plain dict or None."""
    return _execute(
        "SELECT id, full_name, email, created_at FROM users WHERE email = ?",
        (email.strip().lower(),),
        fetch_one=True
    )

def verify_login(email: str, password: str) -> dict | None:
    """Check email + password against the database. Returns user dict or None."""
    row = _execute(
        "SELECT id, full_name, email, password_hash, created_at FROM users WHERE email = ?",
        (email.strip().lower(),),
        fetch_one=True
    )
    if row and check_password_hash(row["password_hash"], password):
        return {
            "id": row["id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "created_at": row["created_at"]
        }
    return None

def get_user_count() -> int:
    """Return total number of registered users."""
    row = _execute("SELECT COUNT(*) AS cnt FROM users", fetch_one=True)
    return row["cnt"] if row else 0
