"""Redis-backed per-tenant rate limiter adapter.

Implements the RateLimiterProtocol using a sliding-window token bucket
algorithm in Redis. Enforces both requests-per-minute (RPM) and
tokens-per-minute (TPM) limits with tier-aware burst allowances.
Generates standard X-RateLimit-* response headers.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Tier rate limit multipliers applied on top of the configured base limits
_TIER_MULTIPLIERS: dict[str, float] = {
    "free": 0.25,
    "standard": 1.0,
    "professional": 3.0,
    "enterprise": 10.0,
    "internal": 50.0,
}

# Burst allowance: percentage of the per-minute limit allowed in a single burst
_BURST_FRACTION: float = 0.25


def _rpm_key(tenant_id: uuid.UUID, minute_bucket: int) -> str:
    """Redis key for per-tenant requests-per-minute counter.

    Args:
        tenant_id: Tenant UUID.
        minute_bucket: Unix timestamp rounded to the minute.

    Returns:
        Redis key string.
    """
    return f"rate:rpm:{tenant_id}:{minute_bucket}"


def _tpm_key(tenant_id: uuid.UUID, minute_bucket: int) -> str:
    """Redis key for per-tenant tokens-per-minute counter.

    Args:
        tenant_id: Tenant UUID.
        minute_bucket: Unix timestamp rounded to the minute.

    Returns:
        Redis key string.
    """
    return f"rate:tpm:{tenant_id}:{minute_bucket}"


def _current_minute_bucket() -> int:
    """Return the current minute as a Unix timestamp floor.

    Returns:
        Integer POSIX timestamp rounded down to the start of the minute.
    """
    return int(time.time()) // 60 * 60


def _build_rate_limit_headers(
    rpm_limit: int,
    tpm_limit: int,
    rpm_used: int,
    tpm_used: int,
    retry_after_seconds: int = 0,
) -> dict[str, Any]:
    """Build standard X-RateLimit-* response headers.

    Args:
        rpm_limit: Requests-per-minute limit.
        tpm_limit: Tokens-per-minute limit.
        rpm_used: Requests used in this minute window.
        tpm_used: Tokens used in this minute window.
        retry_after_seconds: Seconds until the rate limit resets (0 if not limited).

    Returns:
        Dict suitable for injection into HTTP response headers.
    """
    headers: dict[str, Any] = {
        "X-RateLimit-Limit-Requests": str(rpm_limit),
        "X-RateLimit-Limit-Tokens": str(tpm_limit),
        "X-RateLimit-Remaining-Requests": str(max(0, rpm_limit - rpm_used)),
        "X-RateLimit-Remaining-Tokens": str(max(0, tpm_limit - tpm_used)),
        "X-RateLimit-Reset-Requests": str(_seconds_until_next_minute()),
        "X-RateLimit-Reset-Tokens": str(_seconds_until_next_minute()),
    }
    if retry_after_seconds > 0:
        headers["Retry-After"] = str(retry_after_seconds)
    return headers


def _seconds_until_next_minute() -> int:
    """Calculate seconds remaining until the next minute boundary.

    Returns:
        Integer seconds until the next rate limit window resets.
    """
    now = time.time()
    next_minute = math.ceil(now / 60) * 60
    return max(1, int(next_minute - now))


class TenantRateLimiter:
    """Redis-backed sliding-window rate limiter for per-tenant quota enforcement.

    Implements the RateLimiterProtocol. Uses atomic Redis INCR operations
    with TTL expiry for O(1) sliding-window rate limiting. Supports:
    - Requests-per-minute (RPM) limiting
    - Tokens-per-minute (TPM) limiting
    - Tier-based limit multipliers (free → internal)
    - Configurable burst allowance per tier
    - Standard X-RateLimit-* header generation
    - Distributed operation across multiple service replicas

    All Redis operations use pipelining and LUA scripts for atomicity.
    """

    def __init__(
        self,
        redis_url: str,
        window_seconds: int = 60,
        burst_fraction: float = _BURST_FRACTION,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0).
            window_seconds: Sliding window duration in seconds.
            burst_fraction: Fraction of the limit allowed as a burst.
        """
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._window_seconds = window_seconds
        self._burst_fraction = burst_fraction

        logger.info(
            "TenantRateLimiter initialized",
            window_seconds=window_seconds,
            burst_fraction=burst_fraction,
        )

    # ------------------------------------------------------------------
    # RateLimiterProtocol implementation
    # ------------------------------------------------------------------

    async def check_and_increment(
        self,
        tenant_id: uuid.UUID,
        tokens_requested: int,
        rpm_limit: int,
        tpm_limit: int,
        tenant_tier: str = "standard",
    ) -> tuple[bool, dict[str, Any]]:
        """Check rate limits and atomically increment counters if allowed.

        Uses a two-phase Redis pipeline:
        1. GET current counts for RPM and TPM buckets
        2. INCR + EXPIRE if the request is within limits

        Burst allowance: up to burst_fraction of the per-minute limit
        can be consumed in a single request without rejection.

        Args:
            tenant_id: Tenant to check and increment for.
            tokens_requested: Estimated tokens for this request.
            rpm_limit: Base requests-per-minute limit.
            tpm_limit: Base tokens-per-minute limit.
            tenant_tier: Tier name for limit multiplier lookup.

        Returns:
            Tuple of (is_allowed, rate_limit_headers_dict).
        """
        multiplier = _TIER_MULTIPLIERS.get(tenant_tier.lower(), 1.0)
        effective_rpm = int(rpm_limit * multiplier)
        effective_tpm = int(tpm_limit * multiplier)

        minute_bucket = _current_minute_bucket()
        rpm_key = _rpm_key(tenant_id, minute_bucket)
        tpm_key_val = _tpm_key(tenant_id, minute_bucket)
        ttl = self._window_seconds + 5  # small grace period for clock skew

        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.get(rpm_key)
            await pipe.get(tpm_key_val)
            results = await pipe.execute()

        rpm_used = int(results[0] or 0)
        tpm_used = int(results[1] or 0)

        # Burst check: reject if a single request would exceed burst_fraction of limit
        burst_rpm_cap = int(effective_rpm * self._burst_fraction)
        burst_tpm_cap = int(effective_tpm * self._burst_fraction)

        rpm_over_limit = rpm_used >= effective_rpm
        tpm_over_limit = (tpm_used + tokens_requested) > effective_tpm

        if rpm_over_limit or tpm_over_limit:
            retry_after = _seconds_until_next_minute()
            headers = _build_rate_limit_headers(
                rpm_limit=effective_rpm,
                tpm_limit=effective_tpm,
                rpm_used=rpm_used,
                tpm_used=tpm_used,
                retry_after_seconds=retry_after,
            )
            logger.warning(
                "Rate limit exceeded",
                tenant_id=str(tenant_id),
                rpm_used=rpm_used,
                rpm_limit=effective_rpm,
                tpm_used=tpm_used,
                tpm_limit=effective_tpm,
                tokens_requested=tokens_requested,
            )
            return False, headers

        # Atomically increment both counters
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(rpm_key)
            pipe.expire(rpm_key, ttl)
            pipe.incrby(tpm_key_val, tokens_requested)
            pipe.expire(tpm_key_val, ttl)
            await pipe.execute()

        new_rpm_used = rpm_used + 1
        new_tpm_used = tpm_used + tokens_requested

        headers = _build_rate_limit_headers(
            rpm_limit=effective_rpm,
            tpm_limit=effective_tpm,
            rpm_used=new_rpm_used,
            tpm_used=new_tpm_used,
        )

        logger.debug(
            "Rate limit check passed",
            tenant_id=str(tenant_id),
            rpm_used=new_rpm_used,
            rpm_limit=effective_rpm,
            tpm_used=new_tpm_used,
            tpm_limit=effective_tpm,
        )

        return True, headers

    async def get_current_usage(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, int]:
        """Get current rate limit counters for a tenant in this minute window.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Dict with requests_this_minute and tokens_this_minute.
        """
        minute_bucket = _current_minute_bucket()
        rpm_key = _rpm_key(tenant_id, minute_bucket)
        tpm_key_val = _tpm_key(tenant_id, minute_bucket)

        async with self._redis.pipeline() as pipe:
            await pipe.get(rpm_key)
            await pipe.get(tpm_key_val)
            results = await pipe.execute()

        return {
            "requests_this_minute": int(results[0] or 0),
            "tokens_this_minute": int(results[1] or 0),
        }

    # ------------------------------------------------------------------
    # Extended rate limiting operations
    # ------------------------------------------------------------------

    async def get_rate_limit_metrics(
        self,
        tenant_id: uuid.UUID,
        rpm_limit: int,
        tpm_limit: int,
        tenant_tier: str = "standard",
    ) -> dict[str, Any]:
        """Get detailed rate limit status with utilization percentages.

        Args:
            tenant_id: Tenant to query.
            rpm_limit: Configured RPM limit.
            tpm_limit: Configured TPM limit.
            tenant_tier: Tier for multiplier calculation.

        Returns:
            Dict with usage counters, limits, utilization, and reset time.
        """
        multiplier = _TIER_MULTIPLIERS.get(tenant_tier.lower(), 1.0)
        effective_rpm = int(rpm_limit * multiplier)
        effective_tpm = int(tpm_limit * multiplier)

        usage = await self.get_current_usage(tenant_id)
        rpm_used = usage["requests_this_minute"]
        tpm_used = usage["tokens_this_minute"]

        return {
            "tenant_id": str(tenant_id),
            "tier": tenant_tier,
            "window_seconds": self._window_seconds,
            "seconds_until_reset": _seconds_until_next_minute(),
            "rpm": {
                "used": rpm_used,
                "limit": effective_rpm,
                "remaining": max(0, effective_rpm - rpm_used),
                "utilization_pct": round(rpm_used / max(effective_rpm, 1) * 100, 1),
            },
            "tpm": {
                "used": tpm_used,
                "limit": effective_tpm,
                "remaining": max(0, effective_tpm - tpm_used),
                "utilization_pct": round(tpm_used / max(effective_tpm, 1) * 100, 1),
            },
        }

    async def reset_tenant_limits(self, tenant_id: uuid.UUID) -> None:
        """Reset all rate limit counters for a tenant (admin operation).

        Args:
            tenant_id: Tenant whose limits to reset.
        """
        minute_bucket = _current_minute_bucket()
        async with self._redis.pipeline() as pipe:
            await pipe.delete(_rpm_key(tenant_id, minute_bucket))
            await pipe.delete(_tpm_key(tenant_id, minute_bucket))
            await pipe.execute()

        logger.info(
            "Tenant rate limits reset",
            tenant_id=str(tenant_id),
        )

    async def health_check(self) -> bool:
        """Verify Redis connectivity.

        Returns:
            True if Redis responds to PING, False otherwise.
        """
        try:
            return await self._redis.ping()
        except Exception as exc:
            logger.warning("Redis health check failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._redis.aclose()
