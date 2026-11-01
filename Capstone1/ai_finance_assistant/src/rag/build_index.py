"""Rebuild the search index: python -m src.rag.build_index"""
from src.rag.store import build_default_index

if __name__ == "__main__":
    articles, chunks = build_default_index()
    print(f"Indexed {articles} articles into {chunks} chunks")
