"""
rag/retriever.py — Retrieve the top-k most similar resume chunks per JD requirement.

For each requirement sentence extracted from the JD, we embed the sentence
and query ChromaDB for the nearest resume chunks using cosine similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chromadb

from config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved resume chunk with its metadata and similarity score."""

    chunk_id: str
    text: str
    section: str
    similarity: float  # cosine similarity in [0, 1]


@dataclass
class RequirementMatch:
    """Top-k matching resume chunks for a single JD requirement sentence."""

    requirement: str
    matches: list[RetrievedChunk]

    @property
    def best_similarity(self) -> float:
        """Return the highest similarity score among the retrieved chunks."""
        return self.matches[0].similarity if self.matches else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_for_requirements(
    requirements: list[str],
    collection: chromadb.Collection,
    embedding_model: Any,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[RequirementMatch]:
    """For each JD requirement, retrieve the top-k matching resume chunks.

    Args:
        requirements: List of requirement sentences from the JD.
        collection: Populated ChromaDB collection returned by :func:`build_index`.
        embedding_model: A SentenceTransformer model instance.
        top_k: Number of chunks to retrieve per requirement.

    Returns:
        List of :class:`RequirementMatch` objects, one per requirement.
    """
    if not requirements:
        return []

    logger.info("Retrieving chunks for %d JD requirements (top-%d) …", len(requirements), top_k)

    # Batch-embed all requirements at once for efficiency
    embeddings = embedding_model.encode(requirements, show_progress_bar=False).tolist()

    results: list[RequirementMatch] = []
    for req_text, req_embedding in zip(requirements, embeddings):
        chunks = _query_collection(collection, req_embedding, top_k)
        results.append(RequirementMatch(requirement=req_text, matches=chunks))

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _query_collection(
    collection: chromadb.Collection,
    embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Query ChromaDB with a pre-computed embedding and return chunks.

    Args:
        collection: The ChromaDB collection to query.
        embedding: Query embedding vector.
        top_k: Number of results to retrieve.

    Returns:
        List of :class:`RetrievedChunk`, sorted by descending similarity.
    """
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    response = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    docs = response.get("documents", [[]])[0]
    metas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance is in [0, 2] where 0 = identical.
        # Convert to similarity in [0, 1].
        similarity = max(0.0, 1.0 - dist / 2.0) if dist is not None else 0.0
        chunks.append(
            RetrievedChunk(
                chunk_id=str(meta.get("chunk_id", "")),
                text=doc,
                section=str(meta.get("section", "unknown")),
                similarity=round(similarity, 4),
            )
        )

    return chunks

