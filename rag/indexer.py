"""
rag/indexer.py — Chunk resume sections, embed them, and store in ChromaDB.

Each chunk is at most CHUNK_MAX_WORDS words long.  Metadata stored per chunk:
  - section: section label (e.g. "experience")
  - chunk_id: zero-based index within the collection
  - source: "resume"
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

import chromadb

from config import CHUNK_MAX_WORDS, CHROMA_COLLECTION, CHROMA_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    sections: dict[str, str],
    embedding_model: Any,
    collection_name: str = CHROMA_COLLECTION,
    persist_dir: str | None = str(CHROMA_DIR),
    reset: bool = True,
) -> chromadb.Collection:
    """Chunk resume sections, embed, and store in a ChromaDB collection.

    Args:
        sections: Dict mapping section label → section text.
        embedding_model: A SentenceTransformer model instance.
        collection_name: ChromaDB collection name.
        persist_dir: Directory for persistent ChromaDB storage.
                     Pass ``None`` to use an in-memory client.
        reset: If ``True``, delete and recreate the collection on each run
               (avoids stale chunks from previous resumes).

    Returns:
        The populated ChromaDB ``Collection`` object.
    """
    client = _get_client(persist_dir)

    if reset:
        _reset_collection(client, collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks, metadatas, ids = [], [], []
    chunk_counter = 0

    for section_label, section_text in sections.items():
        section_chunks = _chunk_text(section_text, max_words=CHUNK_MAX_WORDS)
        for chunk_text in section_chunks:
            if not chunk_text.strip():
                continue
            chunks.append(chunk_text)
            metadatas.append(
                {
                    "section": section_label,
                    "chunk_id": chunk_counter,
                    "source": "resume",
                }
            )
            ids.append(f"chunk_{chunk_counter}_{uuid.uuid4().hex[:8]}")
            chunk_counter += 1

    if not chunks:
        logger.warning("No chunks to index.")
        return collection

    logger.info("Embedding %d chunks …", len(chunks))
    embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info("Indexed %d chunks into collection '%s'.", len(chunks), collection_name)
    return collection


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Split text into chunks, respecting bullet-point boundaries.

    Strategy:
    1. Split on blank lines and bullet markers to get natural segments.
    2. If a segment exceeds *max_words*, further split it on sentence endings.
    3. Sliding-window merge: accumulate segments until the word count would
       exceed *max_words*, then emit.

    Args:
        text: Section text.
        max_words: Maximum words per chunk.

    Returns:
        List of chunk strings.
    """
    # Split on bullets / blank lines first
    raw_segments = re.split(r"\n\s*[\-\*\u2022]|\n\n+", text)
    raw_segments = [s.strip() for s in raw_segments if s.strip()]

    refined: list[str] = []
    for seg in raw_segments:
        if _word_count(seg) <= max_words:
            refined.append(seg)
        else:
            # Break on sentence endings
            sentences = re.split(r"(?<=[.!?])\s+", seg)
            buf: list[str] = []
            buf_words = 0
            for sent in sentences:
                w = _word_count(sent)
                if buf_words + w > max_words and buf:
                    refined.append(" ".join(buf))
                    buf, buf_words = [], 0
                buf.append(sent)
                buf_words += w
            if buf:
                refined.append(" ".join(buf))

    return refined


def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_client(persist_dir: str | None) -> chromadb.ClientAPI:
    """Return a ChromaDB client (persistent or ephemeral)."""
    if persist_dir:
        return chromadb.PersistentClient(path=persist_dir)
    return chromadb.EphemeralClient()


def _reset_collection(client: chromadb.ClientAPI, name: str) -> None:
    """Delete a collection if it exists (idempotent)."""
    try:
        client.delete_collection(name)
        logger.debug("Deleted existing collection '%s'.", name)
    except Exception:
        pass  # Collection didn't exist — that's fine

