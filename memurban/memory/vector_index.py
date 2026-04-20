"""Local vector index for episodic memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .embeddings import DeterministicTextEmbedder


@dataclass
class SearchHit:
    """Single retrieval hit with score and metadata."""

    score: float
    item: Dict[str, Any]
    metadata: Dict[str, Any]


class InMemoryVectorIndex:
    """Numpy-backed cosine similarity index."""

    def __init__(self, embedder: Optional[DeterministicTextEmbedder] = None) -> None:
        self.embedder = embedder or DeterministicTextEmbedder()
        self.items: List[Dict[str, Any]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.vectors: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, text: str, item: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.texts.append(text)
        self.items.append(item)
        self.metadata.append(metadata or {})
        self.vectors.append(self.embedder.embed(text))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        if not self.vectors:
            return []

        query_vector = self.embedder.embed(query)
        hits: List[SearchHit] = []

        for idx, vector in enumerate(self.vectors):
            metadata = self.metadata[idx]
            if metadata_filter and any(metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            score = float(np.dot(query_vector, vector))
            if score < min_score:
                continue
            hits.append(SearchHit(score=score, item=self.items[idx], metadata=metadata))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def export_state(self) -> Dict[str, Any]:
        return {
            "size": len(self.items),
            "embedding_dim": self.embedder.dim,
            "texts": self.texts,
            "metadata": self.metadata,
        }
