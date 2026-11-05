"""User accounts, conversation ownership, and the conversation history shown in the UI.

LangGraph's checkpoints hold what the agents remember. They don't record which user
owns a thread or which sources and ratings each answer had, so those live here.
Every thread read and write is filtered by user, so knowing a thread ID grants nothing.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import PROJECT_ROOT, get_config

# One of OWASP's recommended scrypt settings. Uses about 64 MB per hash.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 3
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 1024  # stops very long inputs from wasting hashing time
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    profile TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS threads_by_user ON threads(user_id, updated_at);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    caption TEXT,
    agents TEXT NOT NULL DEFAULT '[]',
    rating INTEGER
);
CREATE INDEX IF NOT EXISTS turns_by_thread ON turns(thread_id, id);
"""


class AccountError(Exception):
    """A problem with sign-up details, safe to show to the user."""


def get_db_path() -> Path:
    """FINANCE_DB_PATH overrides the config, which keeps tests away from real data."""
    return Path(os.getenv("FINANCE_DB_PATH") or PROJECT_ROOT / get_config()["database"]["path"])


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    _, n, r, p, salt, expected = stored.split("$")
    derived = _scrypt(password, bytes.fromhex(salt), int(n), int(r), int(p))
    return hmac.compare_digest(derived, bytes.fromhex(expected))


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                          maxmem=128 * n * r * (p + 2), dklen=32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Checked when an email is unknown, so that case takes as long as a wrong password.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


class AccountStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else get_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Owner-only from the start. SQLite gives its extra files the same permissions.
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # off by default, and cascades need it
        return conn

    # --- Accounts ---
    def create_user(self, email: str, password: str) -> dict:
        email = email.strip().lower()
        if not EMAIL_PATTERN.fullmatch(email) or len(email) > 254:
            raise AccountError("Please enter a valid email address.")
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise AccountError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        user = {"id": str(uuid.uuid4()), "email": email}
        password_hash = hash_password(password)
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user["id"], email, password_hash, _now()),
                )
        except sqlite3.IntegrityError:
            raise AccountError("An account with that email already exists.") from None
        return user

    def authenticate(self, email: str, password: str) -> dict | None:
        """Returns the user, or None without revealing whether the email exists."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        if len(password) > MAX_PASSWORD_LENGTH:
            return None
        matches = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        return {"id": row["id"], "email": row["email"]} if row and matches else None

    def get_profile(self, user_id: str) -> dict:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT profile FROM users WHERE id = ?", (user_id,)).fetchone()
        return json.loads(row["profile"]) if row else {}

    def save_profile(self, user_id: str, profile: dict) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE users SET profile = ? WHERE id = ?", (json.dumps(profile), user_id))

    # --- Conversations ---
    def ensure_thread(self, user_id: str, thread_id: str, title: str) -> None:
        """Creates the thread on first use. Refuses a thread that another user owns."""
        now = _now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO threads (id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, title.strip()[:60] or "New conversation", now, now),
            )
            updated = conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ? AND user_id = ?",
                (now, thread_id, user_id),
            ).rowcount
        if not updated:
            raise PermissionError("This conversation belongs to another user.")

    def owns_thread(self, user_id: str, thread_id: str) -> bool:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT 1 FROM threads WHERE id = ? AND user_id = ?", (thread_id, user_id)
            ).fetchone() is not None

    def list_threads(self, user_id: str) -> list[dict]:
        """Most recently used first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, title, updated_at FROM threads WHERE user_id = ? "
                "ORDER BY updated_at DESC, id",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        """Returns whether anything was deleted. The caller also clears its checkpoints."""
        with closing(self._connect()) as conn, conn:
            return conn.execute(
                "DELETE FROM threads WHERE id = ? AND user_id = ?", (thread_id, user_id)
            ).rowcount > 0

    # --- Turns ---
    def add_turn(self, user_id: str, thread_id: str, role: str, content: str,
                 sources: list[str] | None = None, caption: str | None = None,
                 agents: list[str] | None = None) -> None:
        if not self.owns_thread(user_id, thread_id):
            raise PermissionError("This conversation belongs to another user.")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO turns (thread_id, role, content, sources, caption, agents) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, role, content, json.dumps(sources or []), caption, json.dumps(agents or [])),
            )

    def get_turns(self, user_id: str, thread_id: str) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT t.id, t.role, t.content, t.sources, t.caption, t.agents, t.rating "
                "FROM turns t JOIN threads th ON th.id = t.thread_id "
                "WHERE t.thread_id = ? AND th.user_id = ? ORDER BY t.id",
                (thread_id, user_id),
            ).fetchall()
        return [{**dict(row), "sources": json.loads(row["sources"]), "agents": json.loads(row["agents"])}
                for row in rows]

    def rate_turn(self, user_id: str, turn_id: int, rating: int) -> bool:
        """Saves the first rating only. Returns whether this call saved it."""
        with closing(self._connect()) as conn, conn:
            return conn.execute(
                "UPDATE turns SET rating = ? WHERE id = ? AND rating IS NULL AND thread_id IN "
                "(SELECT id FROM threads WHERE user_id = ?)",
                (rating, turn_id, user_id),
            ).rowcount > 0
