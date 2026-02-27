"""Batch scheduler adapter.

Implements dynamic request batching for LLM inference, grouping incoming
requests to maximize GPU utilization. Supports tenant-tier priority queues,
prompt coalescing for identical inputs, throughput tracking, and queue-depth
alerting.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class TenantTier(int, enum.Enum):
    """Tenant service tiers mapped to scheduling priority.

    Higher values are processed first (enterprise before free).
    """

    FREE = 1
    STANDARD = 2
    PROFESSIONAL = 3
    ENTERPRISE = 4
    INTERNAL = 5


@dataclass
class BatchRequest:
    """Single request awaiting inclusion in a batch.

    Attributes:
        request_id: Unique identifier for this request.
        tenant_id: Owning tenant UUID.
        tenant_tier: Tier used for priority scheduling.
        model: Model identifier.
        payload: Serialized request payload (dict).
        prompt_hash: SHA-256 hash of the prompt for coalescing.
        enqueued_at: Monotonic timestamp when the request entered the queue.
        future: asyncio.Future that resolves to the inference result.
    """

    request_id: uuid.UUID
    tenant_id: uuid.UUID
    tenant_tier: TenantTier
    model: str
    payload: dict[str, Any]
    prompt_hash: str
    enqueued_at: float
    future: asyncio.Future[Any] = field(default_factory=asyncio.Future)


@dataclass
class BatchResult:
    """Outcome of executing a single batch.

    Attributes:
        batch_id: Unique batch identifier.
        model: Model used for inference.
        request_count: Number of requests in the batch.
        tokens_in: Total input tokens across all requests.
        tokens_out: Total output tokens across all requests.
        latency_ms: Wall-clock time to execute the batch in milliseconds.
        coalesced_count: Number of requests resolved via prompt coalescing.
        error_count: Number of requests that failed.
    """

    batch_id: uuid.UUID
    model: str
    request_count: int
    tokens_in: int
    tokens_out: int
    latency_ms: int
    coalesced_count: int
    error_count: int


InferenceFn = Callable[[str, list[dict[str, Any]]], Coroutine[Any, Any, list[Any]]]


def _compute_prompt_hash(payload: dict[str, Any]) -> str:
    """Compute a deterministic hash for prompt coalescing.

    Args:
        payload: Request payload dict.

    Returns:
        Hex SHA-256 digest of the canonical prompt representation.
    """
    import json
    # Include model + temperature in the hash to avoid cross-model coalescing
    canonical = json.dumps(
        {
            "model": payload.get("model", ""),
            "messages": payload.get("messages", payload.get("prompt", "")),
            "temperature": payload.get("temperature", 1.0),
            "max_tokens": payload.get("max_tokens"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class BatchScheduler:
    """Dynamic LLM request batcher with tenant-tier priority scheduling.

    Aggregates incoming inference requests into batches to maximize GPU
    throughput. Supports:
    - Dynamic batching: fills batches up to max_batch_size or flush_timeout_ms
    - Priority scheduling: enterprise tenants drain the queue before free-tier
    - Prompt coalescing: identical prompts resolve to one inference call
    - Throughput and latency metrics per model
    - Queue-depth monitoring with configurable alert thresholds

    Usage:
        scheduler = BatchScheduler(inference_fn=my_runner, max_batch_size=8)
        await scheduler.start()
        result = await scheduler.submit(request)
        await scheduler.stop()
    """

    def __init__(
        self,
        inference_fn: InferenceFn,
        max_batch_size: int = 8,
        flush_timeout_ms: int = 50,
        max_queue_depth: int = 1000,
        queue_depth_alert_threshold: int = 500,
        optimal_batch_size_by_model: dict[str, int] | None = None,
    ) -> None:
        """Initialize the batch scheduler.

        Args:
            inference_fn: Async callable that executes a batch:
                          (model, [payload, ...]) → [result, ...]
            max_batch_size: Hard upper bound on batch size.
            flush_timeout_ms: Maximum wait before flushing an incomplete batch.
            max_queue_depth: Maximum number of pending requests before rejecting.
            queue_depth_alert_threshold: Queue depth that triggers an alert log.
            optimal_batch_size_by_model: Per-model batch size overrides.
        """
        self._inference_fn = inference_fn
        self._max_batch_size = max_batch_size
        self._flush_timeout_ms = flush_timeout_ms
        self._max_queue_depth = max_queue_depth
        self._queue_depth_alert_threshold = queue_depth_alert_threshold
        self._optimal_batch_sizes: dict[str, int] = optimal_batch_size_by_model or {}

        # Priority queues per tier (higher priority → processed first)
        self._queues: dict[TenantTier, asyncio.Queue[BatchRequest]] = {
            tier: asyncio.Queue() for tier in TenantTier
        }
        # Coalescing registry: prompt_hash → list of futures waiting on same prompt
        self._coalescing_map: dict[str, list[asyncio.Future[Any]]] = defaultdict(list)

        # Metrics
        self._total_requests: int = 0
        self._total_batches: int = 0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._total_coalesced: int = 0
        self._total_errors: int = 0
        self._latencies_ms: list[float] = []
        self._throughput_tokens_per_sec: float = 0.0

        self._running: bool = False
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background batching worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._batch_worker(), name="batch-scheduler-worker")
        logger.info(
            "BatchScheduler started",
            max_batch_size=self._max_batch_size,
            flush_timeout_ms=self._flush_timeout_ms,
        )

    async def stop(self) -> None:
        """Stop the background worker and drain remaining requests."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped", total_batches_processed=self._total_batches)

    async def submit(
        self,
        tenant_id: uuid.UUID,
        tenant_tier: TenantTier,
        model: str,
        payload: dict[str, Any],
    ) -> Any:
        """Submit a request for batched inference.

        Args:
            tenant_id: Tenant submitting the request.
            tenant_tier: Scheduling priority tier.
            model: Model to use for inference.
            payload: Request payload dict.

        Returns:
            Inference result for this request.

        Raises:
            RuntimeError: If the queue is full or scheduler is not running.
        """
        if not self._running:
            raise RuntimeError("BatchScheduler is not running; call start() first")

        total_depth = sum(q.qsize() for q in self._queues.values())
        if total_depth >= self._max_queue_depth:
            raise RuntimeError(
                f"Batch queue full ({total_depth}/{self._max_queue_depth}); try again later"
            )

        if total_depth >= self._queue_depth_alert_threshold:
            logger.warning(
                "Batch queue depth approaching limit",
                current_depth=total_depth,
                threshold=self._queue_depth_alert_threshold,
                max_depth=self._max_queue_depth,
            )

        prompt_hash = _compute_prompt_hash(payload)
        request = BatchRequest(
            request_id=uuid.uuid4(),
            tenant_id=tenant_id,
            tenant_tier=tenant_tier,
            model=model,
            payload=payload,
            prompt_hash=prompt_hash,
            enqueued_at=time.monotonic(),
        )

        # Coalescing: if an identical prompt is already in flight, reuse its future
        if prompt_hash in self._coalescing_map and self._coalescing_map[prompt_hash]:
            self._total_coalesced += 1
            logger.debug(
                "Coalescing duplicate prompt",
                prompt_hash=prompt_hash[:16],
                tenant_id=str(tenant_id),
            )
            shared_future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            self._coalescing_map[prompt_hash].append(shared_future)
            return await shared_future

        # Register this request as the canonical one for coalescing
        self._coalescing_map[prompt_hash].append(request.future)
        await self._queues[tenant_tier].put(request)
        self._total_requests += 1

        return await request.future

    async def _batch_worker(self) -> None:
        """Main background loop: drain priority queues into batches."""
        while self._running:
            batch = await self._collect_batch()
            if not batch:
                await asyncio.sleep(0.001)
                continue
            await self._execute_batch(batch)

    async def _collect_batch(self) -> list[BatchRequest]:
        """Collect requests from priority queues into one batch.

        Drains higher-priority tiers first, up to max_batch_size. Waits
        up to flush_timeout_ms before returning a partial batch.

        Returns:
            List of BatchRequests to execute as one batch.
        """
        batch: list[BatchRequest] = []
        deadline = time.monotonic() + self._flush_timeout_ms / 1000.0

        while len(batch) < self._max_batch_size and time.monotonic() < deadline:
            request = self._dequeue_by_priority()
            if request is not None:
                batch.append(request)
            else:
                # Nothing available, yield briefly
                await asyncio.sleep(0.001)

        return batch

    def _dequeue_by_priority(self) -> BatchRequest | None:
        """Dequeue one request from the highest-priority non-empty queue.

        Returns:
            Next BatchRequest, or None if all queues are empty.
        """
        for tier in sorted(TenantTier, reverse=True):
            try:
                return self._queues[tier].get_nowait()
            except asyncio.QueueEmpty:
                continue
        return None

    def _get_optimal_batch_size(self, model: str) -> int:
        """Get the optimal batch size for a specific model.

        Args:
            model: Model identifier.

        Returns:
            Optimal batch size.
        """
        return self._optimal_batch_sizes.get(model, self._max_batch_size)

    async def _execute_batch(self, batch: list[BatchRequest]) -> None:
        """Execute a collected batch and resolve all futures.

        Args:
            batch: List of BatchRequests to execute.
        """
        if not batch:
            return

        model = batch[0].model
        batch_id = uuid.uuid4()
        start_time = time.monotonic()

        logger.debug(
            "Executing batch",
            batch_id=str(batch_id),
            model=model,
            batch_size=len(batch),
            wait_times_ms=[
                round((start_time - req.enqueued_at) * 1000, 1) for req in batch
            ],
        )

        payloads = [req.payload for req in batch]

        try:
            results = await self._inference_fn(model, payloads)

            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._latencies_ms.append(latency_ms)
            # Keep only last 1000 latency samples
            if len(self._latencies_ms) > 1000:
                self._latencies_ms = self._latencies_ms[-1000:]

            tokens_in = sum(
                len(str(req.payload.get("messages", req.payload.get("prompt", ""))).split())
                for req in batch
            )
            tokens_out = sum(
                len(str(result).split()) if result else 0 for result in results
            )

            # Update throughput rolling estimate
            elapsed = max((time.monotonic() - start_time), 0.001)
            self._throughput_tokens_per_sec = tokens_out / elapsed

            self._total_batches += 1
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            batch_result = BatchResult(
                batch_id=batch_id,
                model=model,
                request_count=len(batch),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                coalesced_count=0,
                error_count=0,
            )

            logger.info(
                "Batch executed",
                batch_id=str(batch_result.batch_id),
                model=batch_result.model,
                request_count=batch_result.request_count,
                tokens_in=batch_result.tokens_in,
                tokens_out=batch_result.tokens_out,
                latency_ms=batch_result.latency_ms,
                throughput_tps=round(self._throughput_tokens_per_sec, 1),
            )

            # Resolve all futures and clear coalescing entries
            for request, result in zip(batch, results):
                self._resolve_with_coalesced(request, result)

        except Exception as exc:
            self._total_errors += len(batch)
            logger.error(
                "Batch execution failed",
                batch_id=str(batch_id),
                model=model,
                batch_size=len(batch),
                error=str(exc),
            )
            for request in batch:
                self._reject_with_coalesced(request, exc)

    def _resolve_with_coalesced(self, request: BatchRequest, result: Any) -> None:
        """Resolve a request's future and all coalesced waiters.

        Args:
            request: The canonical request that was executed.
            result: Inference result to broadcast.
        """
        prompt_hash = request.prompt_hash
        waiters = self._coalescing_map.pop(prompt_hash, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(result)

    def _reject_with_coalesced(self, request: BatchRequest, exc: Exception) -> None:
        """Reject a request's future and all coalesced waiters with an exception.

        Args:
            request: The canonical request that failed.
            exc: Exception to propagate.
        """
        prompt_hash = request.prompt_hash
        waiters = self._coalescing_map.pop(prompt_hash, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_exception(exc)

    def get_metrics(self) -> dict[str, Any]:
        """Return current throughput and queue metrics.

        Returns:
            Dict containing queue depths, throughput, latency stats,
            and counters for total requests, batches, and errors.
        """
        queue_depths = {tier.name: self._queues[tier].qsize() for tier in TenantTier}
        total_depth = sum(queue_depths.values())

        sorted_latencies = sorted(self._latencies_ms)
        count = len(sorted_latencies)

        def percentile(pct: float) -> float:
            if not sorted_latencies:
                return 0.0
            index = int(count * pct / 100)
            return sorted_latencies[min(index, count - 1)]

        return {
            "queue_depth_total": total_depth,
            "queue_depth_by_tier": queue_depths,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "total_coalesced": self._total_coalesced,
            "total_errors": self._total_errors,
            "throughput_tokens_per_sec": round(self._throughput_tokens_per_sec, 1),
            "batch_latency_p50_ms": round(percentile(50), 1),
            "batch_latency_p95_ms": round(percentile(95), 1),
            "batch_latency_p99_ms": round(percentile(99), 1),
        }
