"""Splits articles into small pieces that can be searched and cited individually."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.articles import Article

_HEADING = re.compile(r"^## +(.+)$", re.MULTILINE)


def _sections(body: str) -> list[tuple[str, str]]:
    """Split on '## ' headings, because a section usually holds one idea."""
    pieces = _HEADING.split(body)  # [intro, heading1, text1, heading2, text2, ...]
    sections = [("", pieces[0].strip())] if pieces[0].strip() else []
    sections += [(h.strip(), t.strip()) for h, t in zip(pieces[1::2], pieces[2::2]) if t.strip()]
    return sections


def chunk_article(article: Article, chunk_size: int = 900, overlap: int = 120) -> list[Document]:
    # Tries paragraph breaks first, then sentences, then words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "]
    )
    documents = []
    for section, text in _sections(article.body):
        context = f"{article.title} - {section}" if section else article.title
        for piece in splitter.split_text(text):
            index = len(documents)
            documents.append(Document(
                id=f"{article.id}:{index}",
                # The context line makes a chunk understandable on its own.
                page_content=f"{context}\n\n{piece}",
                metadata={
                    "article_id": article.id,
                    "title": article.title,
                    "section": section,
                    "category": article.category,
                    "source_name": article.source_name,
                    "source_url": article.source_url,
                    "retrieved": article.retrieved,
                },
            ))
    return documents
