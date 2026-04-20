"""Deterministic local text embeddings for memory retrieval."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

import numpy as np


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text into stable retrieval units."""
    if not text:
        return []

    tokens: List[str] = []
    for raw in TOKEN_PATTERN.findall(text.lower()):
        if len(raw) == 1:
            tokens.append(raw)
            continue
        tokens.append(raw)
        tokens.extend(raw[idx : idx + 2] for idx in range(len(raw) - 1))
    return tokens


class DeterministicTextEmbedder:
    """Hashing-based embedder that runs fully locally and deterministically."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        counts = Counter(tokenize_text(text))
        for token, weight in counts.items():
            vector[hash(token) % self.dim] += float(weight)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def batch_embed(self, texts: Iterable[str]) -> List[np.ndarray]:
        return [self.embed(text) for text in texts]
