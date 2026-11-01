"""Loads settings: config.yaml for tunable values, .env for secrets."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def get_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_openai_key() -> str | None:
    get_config()  # ensures .env has been loaded
    return os.getenv("OPENAI_API_KEY") or None


def get_alpha_vantage_key() -> str | None:
    get_config()
    return os.getenv("ALPHA_VANTAGE_API_KEY") or None
