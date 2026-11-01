"""Keeps answers on the education side of the education-vs-advice line."""
import re

from langchain_core.messages import AIMessage

from src.core.config import get_config
from src.core.state import FinanceState
from src.utils.metrics import annotate, record

ADVICE_PATTERNS = {
    "direct_recommendation": r"\byou should (buy|sell|invest in)\b|\bi recommend (buying|selling)\b",
    "guaranteed_outcome": r"\bguaranteed? (returns?|profits?|gains?)\b|\brisk[- ]free profits?\b",
}
FLAG_NOTE = (
    "**Note:** Part of this answer may read like a specific recommendation. "
    "Treat it as an educational example, not a personal instruction."
)


def check_compliance(text: str) -> list[str]:
    return [
        name for name, pattern in ADVICE_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


def apply_compliance(text: str) -> tuple[str, list[str]]:
    flags = check_compliance(text)
    parts = [text, FLAG_NOTE] if flags else [text]
    parts.append(f"_{get_config()['compliance']['disclaimer']}_")
    return "\n\n".join(parts), flags


def compliance_node(state: FinanceState) -> dict:
    final, flags = apply_compliance(state["draft"])
    annotate(compliance_flags=flags)
    if flags:
        record("compliance_flagged")
    return {"messages": [AIMessage(content=final)], "compliance_flags": flags}
