"""Counts LLM calls, tokens, errors, and finish reasons during a workflow run."""
import threading
from collections import Counter

from langchain_core.callbacks import BaseCallbackHandler

from src.core.config import get_config


class UsageTracker(BaseCallbackHandler):
    """Pass in the run config's callbacks. Agents run in parallel, hence the lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = self.errors = self.input_tokens = self.output_tokens = 0
        self.finish_reasons: Counter = Counter()

    def on_llm_end(self, response, **kwargs) -> None:
        with self._lock:
            self.calls += 1
            for batch in response.generations:
                for generation in batch:
                    info = generation.generation_info or {}
                    self.finish_reasons[info.get("finish_reason", "unknown")] += 1
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)

    def on_llm_error(self, error, **kwargs) -> None:
        with self._lock:
            self.errors += 1

    def cost(self) -> float:
        prices = get_config()["pricing"]
        return (self.input_tokens * prices["input_per_million"]
                + self.output_tokens * prices["output_per_million"]) / 1_000_000
