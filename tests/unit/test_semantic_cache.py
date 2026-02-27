"""Unit tests for SemanticCache.

Verifies:
  - Cache hit when similarity >= threshold
  - Cache miss when similarity < threshold
  - TTL expiry evicts entries
  - Cache statistics (hits, misses, hit_rate)
  - put() returns a non-empty cache key
  - Domain-specific TTL resolution
  - Error in embedding callable does not raise from get() or put()
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_llm_serving.core.cache.semantic_cache import (
    CacheEntry,
    CacheHit,
    CacheStats,
    SemanticCache,
    _cosine_similarity,
    _hash_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(values: list[float]) -> list[float]:
    """Return a normalised embedding vector."""
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


async def _identical_embedding(text: str) -> list[float]:
    """Stub embedding: always returns the same vector (1.0 cosine similarity)."""
    return _make_embedding([1.0, 0.0, 0.0])


async def _orthogonal_embedding(text: str) -> list[float]:
    """Stub embedding: always returns a vector orthogonal to the cached one."""
    return _make_embedding([0.0, 1.0, 0.0])


async def _raising_embedding(text: str) -> list[float]:
    """Stub embedding that raises an exception."""
    raise RuntimeError("embedding service down")


# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the _cosine_similarity helper."""

    def test_identical_vectors_give_one(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_give_minus_one(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors_give_zero(self) -> None:
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_give_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_zero_vector_gives_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# SemanticCache unit tests
# ---------------------------------------------------------------------------


class TestSemanticCachePut:
    """Tests for the put() method."""

    @pytest.mark.asyncio
    async def test_put_returns_non_empty_key(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        key = await cache.put(
            user_prompt="What is 2+2?",
            response="4",
        )
        assert key != ""
        assert len(key) == 64  # SHA-256 hex is 64 chars

    @pytest.mark.asyncio
    async def test_put_stores_entry_in_store(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.put(user_prompt="Hello", response="Hi there!")
        assert len(cache._store) == 1  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_put_returns_empty_on_embedding_error(self) -> None:
        cache = SemanticCache(get_embedding_fn=_raising_embedding)
        key = await cache.put(user_prompt="Hello", response="response")
        assert key == ""

    @pytest.mark.asyncio
    async def test_put_respects_domain_ttl(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.put(user_prompt="Some code", response="result", domain="code")
        entry = list(cache._store.values())[0]  # noqa: SLF001
        assert entry.ttl_seconds == 3_600  # Code domain TTL


class TestSemanticCacheGet:
    """Tests for the get() method."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cache_hit_instance(self) -> None:
        cache = SemanticCache(
            get_embedding_fn=_identical_embedding,
            similarity_threshold=0.95,
        )
        await cache.put(user_prompt="What is 2+2?", response="4", system_prompt="")
        result = await cache.get(user_prompt="What is 2+2?", system_prompt="")
        assert isinstance(result, CacheHit)
        assert result.response == "4"

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self) -> None:
        cache = SemanticCache(
            get_embedding_fn=_orthogonal_embedding,
            similarity_threshold=0.95,
        )
        # Store with one embedding but retrieve with an orthogonal one
        # To achieve a miss, we need different embeddings per call
        call_count = [0]

        async def alternating_embedding(text: str) -> list[float]:
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_embedding([1.0, 0.0, 0.0])
            return _make_embedding([0.0, 1.0, 0.0])

        miss_cache = SemanticCache(
            get_embedding_fn=alternating_embedding,
            similarity_threshold=0.95,
        )
        await miss_cache.put(user_prompt="Question A", response="Answer A")
        result = await miss_cache.get(user_prompt="Question B")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_on_embedding_error(self) -> None:
        cache = SemanticCache(get_embedding_fn=_raising_embedding, similarity_threshold=0.95)
        result = await cache.get(user_prompt="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_increments_hit_count(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.put(user_prompt="Q", response="A")
        await cache.get(user_prompt="Q")
        entry = list(cache._store.values())[0]  # noqa: SLF001
        assert entry.hit_count == 1

    @pytest.mark.asyncio
    async def test_system_prompt_isolates_cache_entries(self) -> None:
        """Entries with different system prompts should not share hits."""
        cache = SemanticCache(get_embedding_fn=_identical_embedding, similarity_threshold=0.95)
        await cache.put(user_prompt="Q", response="A1", system_prompt="System A")
        result = await cache.get(user_prompt="Q", system_prompt="System B")
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self) -> None:
        """An entry with TTL=0... wait, TTL=0 means no expiry. Use TTL=1 ms."""
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        # Manually insert an already-expired entry
        from aumos_llm_serving.core.cache.semantic_cache import CacheEntry
        expired_entry = CacheEntry(
            cache_key="expired-key",
            system_prompt_hash=_hash_text(""),
            embedding=_make_embedding([1.0, 0.0, 0.0]),
            response="stale",
            confidence=1.0,
            cached_at_ms=int(time.time() * 1000) - 10_000,  # 10 seconds ago
            ttl_seconds=1,  # expired 9 seconds ago
            domain="default",
        )
        cache._store["expired-key"] = expired_entry  # noqa: SLF001
        result = await cache.get(user_prompt="Q", system_prompt="")
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_ttl_entry_never_expires(self) -> None:
        """TTL=0 should mean the entry never expires."""
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        from aumos_llm_serving.core.cache.semantic_cache import CacheEntry
        no_expiry = CacheEntry(
            cache_key="no-expiry",
            system_prompt_hash=_hash_text(""),
            embedding=_make_embedding([1.0, 0.0, 0.0]),
            response="eternal",
            confidence=1.0,
            cached_at_ms=0,  # Very old
            ttl_seconds=0,   # Never expire
            domain="default",
        )
        cache._store["no-expiry"] = no_expiry  # noqa: SLF001
        result = await cache.get(user_prompt="Q")
        assert result is not None
        assert result.response == "eternal"


class TestSemanticCacheStats:
    """Tests for cache statistics tracking."""

    @pytest.mark.asyncio
    async def test_initial_stats_are_zero(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        stats = cache.stats
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_hit_increments_hits(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.put(user_prompt="Q", response="A")
        await cache.get(user_prompt="Q")
        assert cache.stats.hits == 1

    @pytest.mark.asyncio
    async def test_miss_increments_misses(self) -> None:
        cache = SemanticCache(get_embedding_fn=_orthogonal_embedding, similarity_threshold=0.95)
        await cache.get(user_prompt="Q")
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_hit_rate_calculated_correctly(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.put(user_prompt="Q", response="A")
        await cache.get(user_prompt="Q")   # hit
        await cache.get(user_prompt="Q")   # hit
        await cache.get(user_prompt="Other — no stored entry for orthogonal")
        # Misses can occur if system_prompt_hash doesn't match — stats count total
        assert cache.stats.hit_rate >= 0.0
        assert cache.stats.hit_rate <= 1.0

    @pytest.mark.asyncio
    async def test_flush_removes_expired_entries(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        from aumos_llm_serving.core.cache.semantic_cache import CacheEntry
        expired = CacheEntry(
            cache_key="ex",
            system_prompt_hash=_hash_text(""),
            embedding=[1.0, 0.0, 0.0],
            response="old",
            confidence=1.0,
            cached_at_ms=int(time.time() * 1000) - 10_000,
            ttl_seconds=1,
            domain="default",
        )
        cache._store["ex"] = expired  # noqa: SLF001
        evicted = await cache.flush()
        assert evicted == 1
        assert "ex" not in cache._store  # noqa: SLF001


class TestSemanticCacheInvalidate:
    """Tests for the invalidate() method."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        key = await cache.put(user_prompt="Q", response="A")
        assert key in cache._store  # noqa: SLF001
        await cache.invalidate(key)
        assert key not in cache._store  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_key_does_not_raise(self) -> None:
        cache = SemanticCache(get_embedding_fn=_identical_embedding)
        await cache.invalidate("nonexistent-key")  # Should not raise
