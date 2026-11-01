"""Thumbs up or down from users, saved locally and marked on a trace."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import PROJECT_ROOT
from src.utils.metrics import record, traced

FEEDBACK_FILE = PROJECT_ROOT / "feedback.jsonl"
_lock = threading.Lock()


def save_feedback(positive: bool, thread_id: str, agents: list[str], path: Path = FEEDBACK_FILE) -> None:
    """Stores the rating and agents only, not the conversation text."""
    entry = {"time": datetime.now(timezone.utc).isoformat(), "positive": positive,
             "thread_id": thread_id, "agents": agents}
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    with traced("user_feedback", "EVALUATOR", positive=positive, agents=agents):
        record("feedback_positive" if positive else "feedback_negative")


def summarize(path: Path = FEEDBACK_FILE) -> dict:
    if not Path(path).exists():
        return {"total": 0, "positive_pct": None}
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    positive = sum(e["positive"] for e in entries)
    return {"total": len(entries), "positive_pct": positive / len(entries) * 100 if entries else None}
