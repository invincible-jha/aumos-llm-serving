"""Semantic similarity-based LLM response cache.

Caches LLM responses keyed by the semantic meaning of the prompt rather than
its exact text. Two prompts that are semantically equivalent (cosine similarity
≥ threshold) share a cached response, dramatically reducing redundant inference
calls for paraphrased or slightly re-worded inputs.

Cache key design:
  key = SHA-256( system_prompt_hash + "|" + base64(embedding) )

Cache value:
  CacheEntry containing the LLM response, confidence score, and timestamps.

Embeddings are provided by an async callable injected at construction time
(dependency injection — no hard dependency on a specific embedding model or
provider). The cache backend is also injected; any async KV store that
implements the simple get/set/delete interface will work.

Usage:
    async def get_embedding(text: str) -> list[float]:
        ...

    cache = SemanticCache(
        get_embedding_fn=get_embedding,
        similarity_threshold=0.95,
    )
    hit = await cache.get(system_prompt="You are a helpful assistant.", user_prompt="What is 2+2?")
    if hit:
        return hit.response
    response = await llm.generate(...)
    await cache.put(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2?",
        response=response,
    )
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Types and DTOs
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A stored LLM response in the semantic cache.

    Attributes:
        cache_key: Unique cache key derived from prompt hashes.
        system_prompt_hash: SHA-256 hex of the system prompt (first 12 chars).
        embedding: Embedding vector of the user prompt.
        response: The LLM response text or structured output.
        confidence: Caller-supplied confidence/quality score in [0.0, 1.0].
        cached_at_ms: Unix epoch milliseconds when this entry was created.
        ttl_seconds: Time-to-live in seconds. 0 means never expires.
        domain: Optional domain label for TTL partitioning (e.g., 'code', 'chat').
        hit_count: Number of times this entry has been served from cache.
    """

    cache_key: str
    system_prompt_hash: str
    embedding: list[float]
    response: Any
    confidence: float
    cached_at_ms: int
    ttl_seconds: int
    domain: str = ""
    hit_count: int = 0


@dataclass
class CacheHit:
    """The result of a successful cache lookup.

    Attributes:
        response: The cached LLM response.
        confidence: Similarity score between the query embedding and the cached entry.
        cached_at_ms: Timestamp when the entry was originally cached.
        cache_key: The cache key that matched.
        similarity: Cosine similarity score that triggered the cache hit.
    """

    response: Any
    confidence: float
    cached_at_ms: int
    cache_key: str
    similarity: float


@dataclass
class CacheStats:
    """Aggregate statistics for the semantic cache.

    Attributes:
        hits: Total number of cache hits.
        misses: Total number of cache misses.
        evictions: Total number of TTL-expired entries removed.
        total_entries: Current number of entries in the cache.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0].

        Returns:
            0.0 if no requests have been made yet.
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 for zero-magnitude vectors.
    """
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash_text(text: str) -> str:
    """Return a hex SHA-256 digest of a text string.

    Args:
        text: Input string to hash.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Default TTL table by domain (seconds)
# ---------------------------------------------------------------------------

DEFAULT_TTL_BY_DOMAIN: dict[str, int] = {
    "code": 3_600,          # 1 hour — code answers may change with library updates
    "chat": 900,            # 15 minutes — conversational context is highly dynamic
    "classification": 7_200,  # 2 hours — label schema changes infrequently
    "summarisation": 1_800,  # 30 minutes
    "translation": 86_400,  # 24 hours — translations are stable
    "default": 1_800,       # 30 minutes fallback
}


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Cache LLM responses by semantic similarity of the prompt.

    Compares each incoming prompt's embedding against stored embeddings using
    cosine similarity. Returns a CacheHit when similarity ≥ threshold and the
    entry has not expired.

    The cache is backed by an in-process dict by default (suitable for single-
    worker deployments). For multi-worker production setups, inject a Redis-backed
    store via the cache_backend parameter.

    Args:
        get_embedding_fn: Async callable that accepts a text string and returns
            a list[float] embedding vector. Signature:
                async (text: str) -> list[float]
        similarity_threshold: Minimum cosine similarity for a cache hit [0.0, 1.0].
            Defaults to 0.95.
        default_ttl_seconds: Default TTL for entries without a domain override.
        ttl_by_domain: Mapping of domain label → TTL seconds. Merged with
            DEFAULT_TTL_BY_DOMAIN.
        cache_backend: Optional async KV backend. If None, an in-process dict is
            used. The backend must expose:
                async get(key: str) -> CacheEntry | None
                async set(key: str, entry: CacheEntry) -> None
                async delete(key: str) -> None
                async keys() -> list[str]
    """

    def __init__(
        self,
        get_embedding_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
        similarity_threshold: float = 0.95,
        default_ttl_seconds: int = 1_800,
        ttl_by_domain: dict[str, int] | None = None,
        cache_backend: Any | None = None,
    ) -> None:
        """Initialise the SemanticCache.

        Args:
            get_embedding_fn: Async embedding callable.
            similarity_threshold: Cosine similarity threshold for cache hits.
            default_ttl_seconds: Default entry TTL in seconds.
            ttl_by_domain: Domain-specific TTL overrides.
            cache_backend: Optional external async KV store.
        """
        self._embed = get_embedding_fn
        self._threshold = similarity_threshold
        self._default_ttl = default_ttl_seconds
        self._ttl_table: dict[str, int] = {**DEFAULT_TTL_BY_DOMAIN, **(ttl_by_domain or {})}
        # In-process store: cache_key → CacheEntry
        self._store: dict[str, CacheEntry] = {}
        self._backend = cache_backend
        self._stats = CacheStats()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_ttl(self, domain: str) -> int:
        """Resolve TTL for a given domain label.

        Args:
            domain: Domain label (e.g. 'code', 'chat').

        Returns:
            TTL in seconds.
        """
        return self._ttl_table.get(domain, self._default_ttl)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check whether a cache entry has exceeded its TTL.

        Args:
            entry: The entry to check.

        Returns:
            True if the entry is expired and should not be served.
        """
        if entry.ttl_seconds == 0:
            return False
        age_seconds = (int(time.time() * 1000) - entry.cached_at_ms) / 1_000
        return age_seconds > entry.ttl_seconds

    async def _backend_get(self, key: str) -> CacheEntry | None:
        """Retrieve an entry from the backend (or in-process store).

        Args:
            key: Cache key.

        Returns:
            CacheEntry or None.
        """
        if self._backend is not None:
            return await self._backend.get(key)
        return self._store.get(key)

    async def _backend_set(self, key: str, entry: CacheEntry) -> None:
        """Store an entry in the backend (or in-process store).

        Args:
            key: Cache key.
            entry: Entry to store.
        """
        if self._backend is not None:
            await self._backend.set(key, entry)
        else:
            self._store[key] = entry

    async def _backend_delete(self, key: str) -> None:
        """Delete an entry from the backend.

        Args:
            key: Cache key to remove.
        """
        if self._backend is not None:
            await self._backend.delete(key)
        else:
            self._store.pop(key, None)

    async def _backend_keys(self) -> list[str]:
        """Return all keys in the backend.

        Returns:
            List of cache keys.
        """
        if self._backend is not None:
            return await self._backend.keys()
        return list(self._store.keys())

    def _build_cache_key(self, system_prompt_hash: str, embedding: list[float]) -> str:
        """Derive a stable cache key from the system prompt hash and embedding.

        Args:
            system_prompt_hash: SHA-256 hex of the system prompt.
            embedding: User-prompt embedding vector.

        Returns:
            Hex SHA-256 of the composite key material.
        """
        # Represent embedding as a rounded string to tolerate tiny float differences
        embedding_repr = ",".join(f"{v:.6f}" for v in embedding)
        composite = f"{system_prompt_hash}|{embedding_repr}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        user_prompt: str,
        system_prompt: str = "",
        domain: str = "default",
    ) -> CacheHit | None:
        """Look up a cached response for the given prompt.

        Computes an embedding for the user prompt, then scans stored entries
        with a matching system-prompt hash for a semantically similar hit.

        Args:
            user_prompt: The user's input text.
            system_prompt: The system prompt (affects response context).
            domain: Domain label used for TTL resolution (e.g. 'code').

        Returns:
            CacheHit if a valid entry is found; None otherwise.
        """
        system_prompt_hash = _hash_text(system_prompt)

        try:
            query_embedding = await self._embed(user_prompt)
        except Exception:
            logger.warning("semantic_cache_embedding_error_on_get", domain=domain)
            self._stats.misses += 1
            return None

        all_keys = await self._backend_keys()

        best_similarity = 0.0
        best_entry: CacheEntry | None = None

        for key in all_keys:
            entry = await self._backend_get(key)
            if entry is None:
                continue
            # Only compare against entries with the same system prompt
            if entry.system_prompt_hash != system_prompt_hash:
                continue
            # Evict expired entries opportunistically
            if self._is_expired(entry):
                await self._backend_delete(key)
                self._stats.evictions += 1
                continue
            similarity = _cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is not None and best_similarity >= self._threshold:
            best_entry.hit_count += 1
            self._stats.hits += 1
            logger.debug(
                "semantic_cache_hit",
                domain=domain,
                similarity=best_similarity,
                cache_key=best_entry.cache_key,
            )
            return CacheHit(
                response=best_entry.response,
                confidence=best_entry.confidence,
                cached_at_ms=best_entry.cached_at_ms,
                cache_key=best_entry.cache_key,
                similarity=best_similarity,
            )

        self._stats.misses += 1
        logger.debug("semantic_cache_miss", domain=domain, best_similarity=best_similarity)
        return None

    async def put(
        self,
        user_prompt: str,
        response: Any,
        system_prompt: str = "",
        confidence: float = 1.0,
        domain: str = "default",
    ) -> str:
        """Store an LLM response in the cache.

        Args:
            user_prompt: The user's input text (used to generate embedding).
            response: The LLM response to cache (any JSON-serialisable type).
            system_prompt: The system prompt that produced this response.
            confidence: Quality/confidence score for this response [0.0, 1.0].
            domain: Domain label for TTL resolution.

        Returns:
            The generated cache key for this entry.
        """
        system_prompt_hash = _hash_text(system_prompt)

        try:
            embedding = await self._embed(user_prompt)
        except Exception:
            logger.warning("semantic_cache_embedding_error_on_put", domain=domain)
            return ""

        ttl = self._resolve_ttl(domain)
        cache_key = self._build_cache_key(system_prompt_hash, embedding)

        entry = CacheEntry(
            cache_key=cache_key,
            system_prompt_hash=system_prompt_hash,
            embedding=embedding,
            response=response,
            confidence=confidence,
            cached_at_ms=int(time.time() * 1000),
            ttl_seconds=ttl,
            domain=domain,
        )

        async with self._lock:
            await self._backend_set(cache_key, entry)

        logger.debug(
            "semantic_cache_stored",
            cache_key=cache_key,
            domain=domain,
            ttl_seconds=ttl,
        )
        return cache_key

    async def invalidate(self, cache_key: str) -> None:
        """Remove a specific entry from the cache.

        Args:
            cache_key: The key returned from a prior put() call.
        """
        async with self._lock:
            await self._backend_delete(cache_key)
        logger.debug("semantic_cache_invalidated", cache_key=cache_key)

    async def flush(self) -> int:
        """Remove all expired entries and return the count of evictions.

        Returns:
            Number of entries removed.
        """
        evicted = 0
        all_keys = await self._backend_keys()
        for key in all_keys:
            entry = await self._backend_get(key)
            if entry is not None and self._is_expired(entry):
                await self._backend_delete(key)
                evicted += 1
        self._stats.evictions += evicted
        logger.info("semantic_cache_flush_complete", evicted=evicted)
        return evicted

    @property
    def stats(self) -> CacheStats:
        """Return current cache statistics.

        Note: total_entries is updated on every stats read for accuracy.

        Returns:
            CacheStats snapshot.
        """
        self._stats.total_entries = len(self._store)
        return self._stats
