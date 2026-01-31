"""
Statement Copilot - Embeddings
==============================
OpenRouter embeddings client with a lightweight local fallback.
Also provides a simple sparse encoder for hybrid search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import List, Dict, Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)


# ------------------------------
# Helpers
# ------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


# ------------------------------
# Dense embeddings
# ------------------------------

class LocalHashEmbeddings:
    """Deterministic local fallback embeddings using a hashing trick."""

    def __init__(self, dimensions: int):
        self.dimensions = max(1, int(dimensions))

    def embed(self, text: str) -> List[float]:
        tokens = _tokenize(text)
        vec = [0.0] * self.dimensions
        if not tokens:
            return vec
        for token in tokens:
            idx = _stable_hash(token) % self.dimensions
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class OpenRouterEmbeddings:
    """OpenRouter embeddings client with local fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.dimensions = int(dimensions or settings.embedding_dimensions)
        self.timeout_s = timeout_s

        self._fallback = LocalHashEmbeddings(self.dimensions)

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. Using local embeddings fallback.")

    def embed(self, text: str) -> List[float]:
        if not self.api_key:
            return self._fallback.embed(text)
        try:
            vectors = self._embed_remote([text])
            return vectors[0] if vectors else self._fallback.embed(text)
        except Exception as exc:
            logger.error(f"OpenRouter embed failed, using fallback: {exc}")
            return self._fallback.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.api_key:
            return self._fallback.embed_batch(texts)
        try:
            vectors = self._embed_remote(texts)
            if len(vectors) != len(texts):
                logger.warning("OpenRouter embeddings count mismatch; using fallback for missing items.")
                # Pad missing vectors with fallback
                fallback = self._fallback.embed_batch(texts)
                for i in range(len(texts)):
                    if i >= len(vectors) or not vectors[i]:
                        vectors.append(fallback[i])
                return vectors[: len(texts)]
            return vectors
        except Exception as exc:
            logger.error(f"OpenRouter embed_batch failed, using fallback: {exc}")
            return self._fallback.embed_batch(texts)

    def _embed_remote(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenRouter connection error: {e}") from e

        latency_ms = int((time.time() - start) * 1000)
        logger.debug(f"OpenRouter embeddings latency={latency_ms}ms, items={len(texts)}")

        data_items = result.get("data", [])
        vectors: List[List[float]] = []
        for item in data_items:
            vec = item.get("embedding")
            if isinstance(vec, list):
                # Ensure L2 normalization for dotproduct metric compatibility
                vectors.append(_l2_normalize(vec))

        if self.dimensions and vectors and len(vectors[0]) != self.dimensions:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vectors[0])}"
            )

        return vectors


# ------------------------------
# Sparse encoder (BM25-like)
# ------------------------------

class SimpleSparseEncoder:
    """Lightweight sparse encoder based on token counts with hashing."""

    def __init__(self, dimension: int = 65536):
        self.dimension = max(1024, int(dimension))

    def encode(self, text: str) -> Dict[str, List]:
        tokens = _tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        counts = Counter(tokens)
        indices = []
        values = []
        for token, count in counts.items():
            idx = _stable_hash(token) % self.dimension
            indices.append(idx)
            values.append(float(count))
        return {"indices": indices, "values": values}


class PineconeBM25Wrapper:
    """Adapter for pinecone_text BM25 encoder with a unified encode() API."""

    def __init__(self):
        from pathlib import Path
        from pinecone_text.sparse import BM25Encoder

        # Load custom-fitted BM25 params if available
        bm25_params_path = Path(__file__).resolve().parents[2] / "db" / "bm25_params.json"

        encoder = BM25Encoder()
        if bm25_params_path.exists():
            encoder.load(str(bm25_params_path))
            logger.info(f"Loaded BM25 params from {bm25_params_path}")
        elif hasattr(encoder, "default"):
            encoder = encoder.default()
            logger.warning("bm25_params.json not found, using default BM25 encoder")

        self.encoder = encoder

    def encode(self, text: str) -> Dict[str, List]:
        if hasattr(self.encoder, "encode_queries"):
            vec = self.encoder.encode_queries(text)
        elif hasattr(self.encoder, "encode"):
            vec = self.encoder.encode(text)
        elif hasattr(self.encoder, "encode_documents"):
            vec = self.encoder.encode_documents([text])[0]
        else:
            raise RuntimeError("Unsupported BM25 encoder API")

        # Some encoders return a list of vectors
        if isinstance(vec, list):
            vec = vec[0] if vec else {"indices": [], "values": []}
        return vec


# ------------------------------
# Singletons
# ------------------------------

_embeddings_client: Optional[OpenRouterEmbeddings] = None
_bm25_encoder: Optional[Any] = None


def get_embeddings_client() -> OpenRouterEmbeddings:
    """Get or create embeddings client singleton."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenRouterEmbeddings()
    return _embeddings_client


def get_bm25_encoder() -> Any:
    """Get or create BM25 encoder singleton."""
    global _bm25_encoder
    if _bm25_encoder is None:
        try:
            _bm25_encoder = PineconeBM25Wrapper()
            logger.info("Using pinecone_text BM25 encoder.")
        except Exception as exc:
            logger.warning(f"pinecone_text not available, using simple sparse encoder: {exc}")
            _bm25_encoder = SimpleSparseEncoder()
    return _bm25_encoder


__all__ = [
    "get_embeddings_client",
    "get_bm25_encoder",
    "OpenRouterEmbeddings",
    "LocalHashEmbeddings",
    "SimpleSparseEncoder",
]