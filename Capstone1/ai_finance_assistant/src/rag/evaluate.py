"""Measure retrieval quality: python -m src.rag.evaluate

Calls OpenAI for each question, so it costs a small amount.
"""
import statistics
import time

import yaml

from src.core.config import PROJECT_ROOT, get_config
from src.rag.store import get_store


def evaluate() -> dict:
    with open(PROJECT_ROOT / "tests/data/retrieval_eval.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    store = get_store()
    k, min_score = get_config()["rag"]["top_k"], get_config()["rag"]["min_score"]
    first = top_k = 0
    latencies, correct_scores, misses = [], [], []
    for item in data["questions"]:
        start = time.perf_counter()
        hits = store.similarity_search_with_score(item["q"], k=k)
        latencies.append(time.perf_counter() - start)
        ids = [doc.metadata["article_id"] for doc, _ in hits]
        if ids[0] in item["expect"]:
            first += 1
        if set(ids) & set(item["expect"]):
            top_k += 1
            correct_scores.append(max(1 - d for doc, d in hits if doc.metadata["article_id"] in item["expect"]))
        else:
            misses.append((item["q"], ids[0]))
    off = {q: 1 - store.similarity_search_with_score(q, k=1)[0][1] for q in data["off_topic"]}
    total = len(data["questions"])
    return {
        "questions": total,
        "hit_at_1_pct": first / total * 100,
        "hit_at_k_pct": top_k / total * 100,
        "k": k,
        "lowest_correct_score": min(correct_scores),
        "correct_below_cutoff": sum(s < min_score for s in correct_scores),
        "highest_off_topic_score": max(off.values()),
        "off_topic_above_cutoff": sum(s >= min_score for s in off.values()),
        "median_latency_ms": statistics.median(latencies) * 1000,
        "misses": misses,
    }


if __name__ == "__main__":
    for name, value in evaluate().items():
        print(f"{name}: {value}")
