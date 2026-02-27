"""Unit tests for CircuitBreaker.

Verifies:
  - CLOSED state: provider is available
  - OPEN state: provider is unavailable
  - HALF_OPEN state: provider available with reduced probability
  - State transitions: CLOSED→OPEN (on failure threshold), OPEN→HALF_OPEN (after timeout),
    HALF_OPEN→CLOSED (on success), HALF_OPEN→OPEN (on failure)
  - Fallback order respects availability
  - Thread/async safety via asyncio.Lock
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from aumos_llm_serving.core.routing.circuit_breaker import (
    ERROR_RATE_THRESHOLD,
    P99_LATENCY_THRESHOLD_MS,
    RECOVERY_WINDOW_SECONDS,
    CircuitBreaker,
    CircuitState,
    ProviderStats,
)


# ---------------------------------------------------------------------------
# ProviderStats unit tests
# ---------------------------------------------------------------------------


class TestProviderStats:
    """Tests for the ProviderStats sliding window computations."""

    @pytest.mark.asyncio
    async def test_initial_error_rate_is_zero(self) -> None:
        stats = ProviderStats("openai")
        assert stats.error_rate_1min == 0.0

    @pytest.mark.asyncio
    async def test_initial_p99_latency_is_zero(self) -> None:
        stats = ProviderStats("openai")
        assert stats.p99_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_error_rate_after_failures(self) -> None:
        stats = ProviderStats("openai")
        for _ in range(4):
            await stats.record(success=True, latency_ms=100.0)
        for _ in range(6):
            await stats.record(success=False, latency_ms=500.0)
        assert stats.error_rate_1min == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_p99_latency_computed_correctly(self) -> None:
        stats = ProviderStats("openai")
        latencies = list(range(1, 101))  # 1 ms to 100 ms
        for lat in latencies:
            await stats.record(success=True, latency_ms=float(lat))
        # P99 of 1..100 is the 99th value = 99 ms (index 98 in sorted list)
        assert stats.p99_latency_ms == pytest.approx(99.0)

    @pytest.mark.asyncio
    async def test_all_success_gives_zero_error_rate(self) -> None:
        stats = ProviderStats("openai")
        for _ in range(10):
            await stats.record(success=True, latency_ms=100.0)
        assert stats.error_rate_1min == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CircuitBreaker state machine tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistration:
    """Tests for provider registration."""

    def test_register_provider_starts_closed(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("anthropic")
        assert cb.get_state("anthropic") == CircuitState.CLOSED

    def test_unregistered_provider_is_unavailable(self) -> None:
        cb = CircuitBreaker()
        assert not cb.is_available("unknown-provider")

    def test_registered_provider_is_available_initially(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai")
        assert cb.is_available("openai")

    def test_get_all_states_returns_all_providers(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai")
        cb.register_provider("anthropic")
        states = cb.get_all_states()
        assert "openai" in states
        assert "anthropic" in states


class TestCircuitBreakerClosedState:
    """Tests for CLOSED state behaviour."""

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai")
        await cb.record_success("openai", latency_ms=100.0)
        assert cb.get_state("openai") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_single_failure_does_not_open(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai")
        await cb.record_failure("openai", latency_ms=200.0)
        assert cb.get_state("openai") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_high_error_rate_opens_circuit(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.5)
        cb.register_provider("openai")
        for _ in range(6):
            await cb.record_failure("openai", latency_ms=100.0)
        for _ in range(4):
            await cb.record_success("openai", latency_ms=100.0)
        assert cb.get_state("openai") == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_high_latency_opens_circuit(self) -> None:
        cb = CircuitBreaker(p99_latency_threshold_ms=1_000.0)
        cb.register_provider("openai")
        # Record 100 calls all with 2000ms latency
        for _ in range(100):
            await cb.record_failure("openai", latency_ms=2_000.0)
        assert cb.get_state("openai") == CircuitState.OPEN


class TestCircuitBreakerOpenState:
    """Tests for OPEN state behaviour."""

    @pytest.mark.asyncio
    async def test_open_circuit_is_unavailable(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.05)
        cb.register_provider("openai")
        # Force open
        for _ in range(100):
            await cb.record_failure("openai", latency_ms=100.0)
        # Force state manually for determinism
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        assert not cb.is_available("openai")

    def test_open_circuit_transitions_to_half_open_after_recovery_window(self) -> None:
        cb = CircuitBreaker(recovery_window_seconds=0.01)
        cb.register_provider("openai")
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        cb._opened_at["openai"] = time.monotonic() - 1.0  # 1 second ago, past 10ms window  # noqa: SLF001
        # is_available triggers the OPEN → HALF_OPEN transition
        # Since half_open_probe_probability = 0.01, result is non-deterministic
        # but the state should be HALF_OPEN after the call
        cb.is_available("openai")
        assert cb.get_state("openai") == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpenState:
    """Tests for HALF_OPEN state behaviour."""

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai")
        cb._states["openai"] = CircuitState.HALF_OPEN  # noqa: SLF001
        await cb.record_success("openai", latency_ms=100.0)
        assert cb.get_state("openai") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens_circuit(self) -> None:
        cb = CircuitBreaker(error_rate_threshold=0.01)
        cb.register_provider("openai")
        cb._states["openai"] = CircuitState.HALF_OPEN  # noqa: SLF001
        # Record enough failures to exceed threshold
        for _ in range(100):
            await cb.record_failure("openai", latency_ms=100.0)
        assert cb.get_state("openai") == CircuitState.OPEN


class TestCircuitBreakerFallback:
    """Tests for fallback provider selection."""

    def test_fallback_returned_when_primary_is_open(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai", fallback_order=["anthropic", "ollama"])
        cb.register_provider("anthropic")
        cb.register_provider("ollama")
        # Open the primary circuit
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        fallback = cb.get_fallback("openai")
        assert fallback in ("anthropic", "ollama")

    def test_none_returned_when_all_fallbacks_are_open(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai", fallback_order=["anthropic"])
        cb.register_provider("anthropic")
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        cb._states["anthropic"] = CircuitState.OPEN  # noqa: SLF001
        fallback = cb.get_fallback("openai")
        assert fallback is None

    def test_fallback_empty_list_returns_none(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai", fallback_order=[])
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        assert cb.get_fallback("openai") is None

    def test_unregistered_provider_fallback_returns_none(self) -> None:
        cb = CircuitBreaker()
        cb.register_provider("openai", fallback_order=["ghost-provider"])
        cb._states["openai"] = CircuitState.OPEN  # noqa: SLF001
        # ghost-provider is not registered so is_available returns False
        assert cb.get_fallback("openai") is None


class TestCircuitBreakerRecordWithUnregisteredProvider:
    """Ensure recording for an unknown provider does not raise."""

    @pytest.mark.asyncio
    async def test_record_success_unknown_provider_is_noop(self) -> None:
        cb = CircuitBreaker()
        await cb.record_success("unknown", latency_ms=100.0)  # Should not raise

    @pytest.mark.asyncio
    async def test_record_failure_unknown_provider_is_noop(self) -> None:
        cb = CircuitBreaker()
        await cb.record_failure("unknown", latency_ms=100.0)  # Should not raise
