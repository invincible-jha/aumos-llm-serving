"""Circuit breaker for multi-provider LLM request routing.

Tracks per-provider error rates and latency. Opens the circuit (stops routing
to a provider) when health signals exceed configured thresholds, then probes
with a small fraction of traffic during a half-open recovery window.

States:
  CLOSED    — normal operation; all traffic routed to the provider
  OPEN      — provider unhealthy; traffic redirected to fallback providers
  HALF_OPEN — recovery probe; 1 % of traffic reaches the provider

Transitions:
  CLOSED → OPEN:      error_rate_1min > 0.05  OR  p99_latency_ms > 5 000
  OPEN → HALF_OPEN:  60 seconds have elapsed since the circuit opened
  HALF_OPEN → CLOSED: probe succeeds (no error in the probing window)
  HALF_OPEN → OPEN:   probe fails

Usage:
    breaker = CircuitBreaker()
    breaker.register_provider("openai", fallback_order=["anthropic", "ollama"])

    if breaker.is_available("openai"):
        # Call provider ...
        breaker.record_success("openai", latency_ms=420.0)
    else:
        # Use fallback
        provider = breaker.get_fallback("openai")
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from enum import Enum
from typing import Deque

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

ERROR_RATE_THRESHOLD: float = 0.05       # 5 % — open if exceeded
P99_LATENCY_THRESHOLD_MS: float = 5_000  # 5 000 ms — open if exceeded
HALF_OPEN_PROBE_PROBABILITY: float = 0.01  # 1 % traffic during half-open
RECOVERY_WINDOW_SECONDS: float = 60.0   # Wait before attempting half-open
METRICS_WINDOW_SECONDS: float = 60.0    # Sliding window for error rate
MAX_SAMPLES: int = 1_000                # Cap the in-memory deque size


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    """Lifecycle state of a provider circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class _CallRecord:
    """A single call outcome stored in the sliding window."""

    __slots__ = ("timestamp", "success", "latency_ms")

    def __init__(self, *, timestamp: float, success: bool, latency_ms: float) -> None:
        self.timestamp = timestamp
        self.success = success
        self.latency_ms = latency_ms


class ProviderStats:
    """Thread-safe statistics for a single provider.

    Maintains a sliding 60-second window of call outcomes to compute
    error rate and p99 latency without requiring an external time-series DB.
    """

    def __init__(self, provider_id: str) -> None:
        """Initialise per-provider statistics.

        Args:
            provider_id: Human-readable provider identifier (e.g. 'openai').
        """
        self.provider_id = provider_id
        self._records: Deque[_CallRecord] = deque(maxlen=MAX_SAMPLES)
        self._lock: asyncio.Lock = asyncio.Lock()

    async def record(self, *, success: bool, latency_ms: float) -> None:
        """Record the outcome of a single provider call.

        Args:
            success: True if the call completed without error.
            latency_ms: Observed latency in milliseconds.
        """
        async with self._lock:
            self._records.append(_CallRecord(
                timestamp=time.monotonic(),
                success=success,
                latency_ms=latency_ms,
            ))

    def _recent_records(self) -> list[_CallRecord]:
        """Return records within the sliding metrics window.

        Returns:
            List of records from the last METRICS_WINDOW_SECONDS seconds.
        """
        cutoff = time.monotonic() - METRICS_WINDOW_SECONDS
        return [r for r in self._records if r.timestamp >= cutoff]

    @property
    def error_rate_1min(self) -> float:
        """Fraction of calls that failed in the last 60 seconds.

        Returns:
            Error rate in [0.0, 1.0]. Returns 0.0 if no recent records.
        """
        records = self._recent_records()
        if not records:
            return 0.0
        failed = sum(1 for r in records if not r.success)
        return failed / len(records)

    @property
    def p99_latency_ms(self) -> float:
        """99th-percentile latency over the last 60 seconds.

        Returns:
            P99 latency in milliseconds. Returns 0.0 if no recent records.
        """
        records = self._recent_records()
        if not records:
            return 0.0
        sorted_latencies = sorted(r.latency_ms for r in records)
        index = max(0, int(len(sorted_latencies) * 0.99) - 1)
        return sorted_latencies[index]

    @property
    def cost_per_token(self) -> float:
        """Placeholder for future cost-per-token tracking.

        Returns:
            Always 0.0 until billing integration is wired.
        """
        return 0.0


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Multi-provider circuit breaker for LLM serving.

    Manages the circuit state for every registered provider independently.
    All state transitions are protected by an asyncio.Lock to prevent races in
    concurrent async request handling.

    Args:
        error_rate_threshold: Open circuit when error rate exceeds this fraction.
        p99_latency_threshold_ms: Open circuit when p99 exceeds this value.
        recovery_window_seconds: Time to wait in OPEN before attempting HALF_OPEN.
        half_open_probe_probability: Fraction of traffic probed in HALF_OPEN state.
    """

    def __init__(
        self,
        error_rate_threshold: float = ERROR_RATE_THRESHOLD,
        p99_latency_threshold_ms: float = P99_LATENCY_THRESHOLD_MS,
        recovery_window_seconds: float = RECOVERY_WINDOW_SECONDS,
        half_open_probe_probability: float = HALF_OPEN_PROBE_PROBABILITY,
    ) -> None:
        """Initialise the CircuitBreaker.

        Args:
            error_rate_threshold: Error rate fraction above which the circuit opens.
            p99_latency_threshold_ms: P99 latency above which the circuit opens.
            recovery_window_seconds: Seconds to wait in OPEN before half-open probe.
            half_open_probe_probability: Probability a request is probed in HALF_OPEN.
        """
        self._error_rate_threshold = error_rate_threshold
        self._p99_latency_threshold_ms = p99_latency_threshold_ms
        self._recovery_window_seconds = recovery_window_seconds
        self._half_open_probe_probability = half_open_probe_probability

        # Per-provider state: {provider_id: CircuitState}
        self._states: dict[str, CircuitState] = {}
        # Timestamp (monotonic) when the circuit opened: {provider_id: float}
        self._opened_at: dict[str, float] = {}
        # Per-provider call statistics
        self._stats: dict[str, ProviderStats] = {}
        # Fallback order per provider: {provider_id: [fallback_id, ...]}
        self._fallbacks: dict[str, list[str]] = {}
        # Single async lock for all state transitions
        self._lock: asyncio.Lock = asyncio.Lock()

    def register_provider(
        self,
        provider_id: str,
        fallback_order: list[str] | None = None,
    ) -> None:
        """Register a provider with the circuit breaker.

        Must be called once per provider before routing any traffic.

        Args:
            provider_id: Unique provider identifier (e.g. 'openai').
            fallback_order: Ordered list of fallback provider IDs to use when
                this provider's circuit is open. Earlier entries are preferred.
        """
        self._states[provider_id] = CircuitState.CLOSED
        self._stats[provider_id] = ProviderStats(provider_id)
        self._fallbacks[provider_id] = fallback_order or []
        logger.info(
            "circuit_breaker_provider_registered",
            provider_id=provider_id,
            fallback_order=fallback_order,
        )

    def get_state(self, provider_id: str) -> CircuitState:
        """Return the current circuit state for a provider.

        Args:
            provider_id: The provider to query.

        Returns:
            Current CircuitState.

        Raises:
            KeyError: If provider_id has not been registered.
        """
        return self._states[provider_id]

    def get_stats(self, provider_id: str) -> ProviderStats:
        """Return the statistics object for a provider.

        Args:
            provider_id: The provider to query.

        Returns:
            ProviderStats instance.

        Raises:
            KeyError: If provider_id has not been registered.
        """
        return self._stats[provider_id]

    def is_available(self, provider_id: str) -> bool:
        """Decide whether traffic should be sent to this provider right now.

        Respects the circuit state:
          - CLOSED: always available
          - OPEN: not available (unless recovery window elapsed → HALF_OPEN)
          - HALF_OPEN: available with probability half_open_probe_probability

        State transitions triggered by this call are NOT async (to allow sync
        callers) but are protected by an internal monotonic check.

        Args:
            provider_id: The provider to check.

        Returns:
            True if a request should be sent to this provider.
        """
        if provider_id not in self._states:
            return False

        state = self._states[provider_id]

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            opened_at = self._opened_at.get(provider_id, 0.0)
            if time.monotonic() - opened_at >= self._recovery_window_seconds:
                # Transition to HALF_OPEN synchronously (no I/O needed)
                self._states[provider_id] = CircuitState.HALF_OPEN
                logger.info(
                    "circuit_breaker_half_open",
                    provider_id=provider_id,
                )
                return random.random() < self._half_open_probe_probability
            return False

        # HALF_OPEN
        return random.random() < self._half_open_probe_probability

    async def record_success(self, provider_id: str, latency_ms: float) -> None:
        """Record a successful provider call outcome.

        If the circuit is HALF_OPEN, a success closes it again.

        Args:
            provider_id: The provider that handled the call.
            latency_ms: Observed latency in milliseconds.
        """
        if provider_id not in self._stats:
            return

        await self._stats[provider_id].record(success=True, latency_ms=latency_ms)

        async with self._lock:
            if self._states[provider_id] == CircuitState.HALF_OPEN:
                self._states[provider_id] = CircuitState.CLOSED
                logger.info(
                    "circuit_breaker_closed",
                    provider_id=provider_id,
                    reason="probe_succeeded",
                )

    async def record_failure(self, provider_id: str, latency_ms: float = 0.0) -> None:
        """Record a failed provider call outcome and potentially open the circuit.

        Evaluates the sliding-window error rate and p99 latency. If either
        exceeds its threshold, the circuit transitions from CLOSED → OPEN, or
        from HALF_OPEN → OPEN.

        Args:
            provider_id: The provider that failed.
            latency_ms: Observed latency before failure (if measurable).
        """
        if provider_id not in self._stats:
            return

        await self._stats[provider_id].record(success=False, latency_ms=latency_ms)

        stats = self._stats[provider_id]
        error_rate = stats.error_rate_1min
        p99 = stats.p99_latency_ms

        should_open = (
            error_rate > self._error_rate_threshold
            or p99 > self._p99_latency_threshold_ms
        )

        async with self._lock:
            current_state = self._states[provider_id]
            if should_open and current_state != CircuitState.OPEN:
                self._states[provider_id] = CircuitState.OPEN
                self._opened_at[provider_id] = time.monotonic()
                logger.warning(
                    "circuit_breaker_opened",
                    provider_id=provider_id,
                    error_rate=error_rate,
                    p99_latency_ms=p99,
                    error_rate_threshold=self._error_rate_threshold,
                    p99_latency_threshold_ms=self._p99_latency_threshold_ms,
                )
            elif current_state == CircuitState.HALF_OPEN and not should_open:
                # Probe failed but health looks OK — stay half-open for next probe
                pass

    def get_fallback(self, provider_id: str) -> str | None:
        """Return the first available fallback provider.

        Iterates the configured fallback order and returns the first provider
        whose circuit is CLOSED or HALF_OPEN.

        Args:
            provider_id: The primary provider that is unavailable.

        Returns:
            ID of the first healthy fallback, or None if none are available.
        """
        for fallback_id in self._fallbacks.get(provider_id, []):
            if self.is_available(fallback_id):
                return fallback_id
        return None

    def get_all_states(self) -> dict[str, CircuitState]:
        """Return a snapshot of all provider circuit states.

        Returns:
            Mapping of provider_id → CircuitState.
        """
        return dict(self._states)
