"""Inference metrics collector adapter.

Tracks throughput, latency distributions, time-to-first-token (TTFT),
inter-token latency (ITL), GPU utilization, queue wait times, and error
rates per model and tenant. Exports Prometheus-compatible metrics.
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Rolling window size for latency samples per model
_MAX_LATENCY_SAMPLES: int = 2000
# Maximum age of a latency sample before it ages out (seconds)
_SAMPLE_MAX_AGE_SECONDS: float = 3600.0


@dataclass
class RequestTrace:
    """Single request lifecycle trace for metrics aggregation.

    Attributes:
        request_id: Unique request identifier.
        tenant_id: Owning tenant.
        model: Model used for inference.
        provider: Provider that served the request.
        enqueued_at: Monotonic timestamp when the request entered the queue.
        started_at: Monotonic timestamp when inference began.
        first_token_at: Monotonic timestamp of the first token (streaming).
        completed_at: Monotonic timestamp when inference completed.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens generated.
        status: Request outcome ('success', 'error', 'rate_limited').
        error_code: Optional error code if status is 'error'.
    """

    request_id: uuid.UUID
    tenant_id: uuid.UUID
    model: str
    provider: str
    enqueued_at: float
    started_at: float
    first_token_at: float | None
    completed_at: float
    prompt_tokens: int
    completion_tokens: int
    status: str
    error_code: str | None = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model.

    Attributes:
        model: Model identifier.
        total_requests: Total requests processed.
        successful_requests: Requests that completed successfully.
        error_requests: Requests that resulted in errors.
        total_prompt_tokens: Cumulative input tokens processed.
        total_completion_tokens: Cumulative output tokens generated.
        latency_samples_ms: Rolling window of end-to-end latency samples (ms).
        ttft_samples_ms: Rolling window of time-to-first-token samples (ms).
        itl_samples_ms: Rolling window of inter-token latency samples (ms).
        queue_wait_samples_ms: Rolling window of queue wait time samples (ms).
        throughput_tokens_per_sec: Exponential moving average of tokens/sec.
    """

    model: str
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latency_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_LATENCY_SAMPLES))
    ttft_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_LATENCY_SAMPLES))
    itl_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_LATENCY_SAMPLES))
    queue_wait_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_LATENCY_SAMPLES))
    throughput_tokens_per_sec: float = 0.0


def _percentile(samples: list[float], pct: float) -> float:
    """Compute an approximate percentile from a sorted sample list.

    Args:
        samples: Sorted list of numeric samples.
        pct: Percentile to compute (0-100).

    Returns:
        Approximate percentile value, or 0.0 if samples is empty.
    """
    if not samples:
        return 0.0
    count = len(samples)
    index = int(math.ceil(count * pct / 100.0)) - 1
    return samples[max(0, min(index, count - 1))]


def _ema(current: float, new_value: float, alpha: float = 0.1) -> float:
    """Compute an exponential moving average update.

    Args:
        current: Previous EMA value.
        new_value: New observation.
        alpha: Smoothing factor (0 < alpha < 1). Higher = more reactive.

    Returns:
        Updated EMA value.
    """
    return alpha * new_value + (1.0 - alpha) * current


class InferenceMetricsCollector:
    """Production-grade inference metrics aggregator.

    Records per-model and per-tenant performance metrics from request traces.
    Computes latency distributions (P50/P95/P99), TTFT, ITL, throughput,
    queue wait times, GPU utilization, and error rates.

    Metrics are available for:
    - Prometheus scraping via get_prometheus_metrics()
    - Dashboard consumption via get_model_summary() and get_tenant_summary()
    - Alert evaluation via get_error_rate() and get_throughput()

    All operations are in-process (no external dependencies). Use alongside
    an OTel exporter for distributed tracing.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector with empty state."""
        # Per-model aggregated metrics
        self._model_metrics: dict[str, ModelMetrics] = defaultdict(
            lambda: ModelMetrics(model="unknown")
        )
        # Per-tenant request counters: {tenant_id: {status: count}}
        self._tenant_counters: dict[uuid.UUID, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # GPU utilization samples per model: {model: deque[(timestamp, pct)]}
        self._gpu_utilization: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=500)
        )
        # Service-wide counters
        self._total_requests: int = 0
        self._started_at: float = time.monotonic()

    def record_request(self, trace: RequestTrace) -> None:
        """Record a completed request trace and update all metrics.

        Args:
            trace: Completed RequestTrace with all timestamps populated.
        """
        self._total_requests += 1

        model_key = trace.model
        if model_key not in self._model_metrics:
            self._model_metrics[model_key] = ModelMetrics(model=model_key)

        metrics = self._model_metrics[model_key]
        metrics.total_requests += 1
        metrics.total_prompt_tokens += trace.prompt_tokens
        metrics.total_completion_tokens += trace.completion_tokens

        # End-to-end latency
        latency_ms = (trace.completed_at - trace.started_at) * 1000.0
        metrics.latency_samples_ms.append(latency_ms)

        # Queue wait time
        queue_wait_ms = (trace.started_at - trace.enqueued_at) * 1000.0
        if queue_wait_ms >= 0:
            metrics.queue_wait_samples_ms.append(queue_wait_ms)

        # Time-to-first-token (only for streaming requests)
        if trace.first_token_at is not None:
            ttft_ms = (trace.first_token_at - trace.started_at) * 1000.0
            if ttft_ms >= 0:
                metrics.ttft_samples_ms.append(ttft_ms)

        # Inter-token latency (approximate from total duration / token count)
        if trace.completion_tokens > 1:
            generation_ms = (trace.completed_at - (trace.first_token_at or trace.started_at)) * 1000.0
            itl_ms = generation_ms / max(trace.completion_tokens - 1, 1)
            metrics.itl_samples_ms.append(itl_ms)

        # Throughput EMA (tokens / elapsed seconds)
        elapsed = max(trace.completed_at - trace.started_at, 0.001)
        tokens_per_sec = trace.completion_tokens / elapsed
        metrics.throughput_tokens_per_sec = _ema(
            metrics.throughput_tokens_per_sec,
            tokens_per_sec,
            alpha=0.2,
        )

        # Status counters
        if trace.status == "success":
            metrics.successful_requests += 1
        else:
            metrics.error_requests += 1

        # Per-tenant counters
        self._tenant_counters[trace.tenant_id][trace.status] += 1

        logger.debug(
            "Request trace recorded",
            model=trace.model,
            provider=trace.provider,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            status=trace.status,
        )

    def record_gpu_utilization(self, model: str, utilization_pct: float) -> None:
        """Record a GPU utilization sample for a model.

        Args:
            model: Model currently loaded on the GPU.
            utilization_pct: GPU utilization percentage (0–100).
        """
        self._gpu_utilization[model].append((time.monotonic(), utilization_pct))

    def get_model_summary(self, model: str) -> dict[str, Any]:
        """Get a complete performance summary for a model.

        Args:
            model: Model identifier.

        Returns:
            Dict with throughput, latency percentiles, TTFT, ITL, error rate,
            token counts, and GPU utilization.
        """
        if model not in self._model_metrics:
            return {"model": model, "total_requests": 0, "status": "no_data"}

        metrics = self._model_metrics[model]
        latencies = sorted(metrics.latency_samples_ms)
        ttft_samples = sorted(metrics.ttft_samples_ms)
        itl_samples = sorted(metrics.itl_samples_ms)
        queue_waits = sorted(metrics.queue_wait_samples_ms)

        # GPU utilization: average of recent samples
        gpu_samples = list(self._gpu_utilization.get(model, []))
        avg_gpu_utilization = (
            statistics.mean(sample[1] for sample in gpu_samples[-100:])
            if gpu_samples
            else 0.0
        )

        error_rate = (
            metrics.error_requests / max(metrics.total_requests, 1) * 100
        )

        return {
            "model": model,
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "error_requests": metrics.error_requests,
            "error_rate_pct": round(error_rate, 2),
            "total_prompt_tokens": metrics.total_prompt_tokens,
            "total_completion_tokens": metrics.total_completion_tokens,
            "throughput_tokens_per_sec": round(metrics.throughput_tokens_per_sec, 1),
            "latency_ms": {
                "p50": round(_percentile(latencies, 50), 1),
                "p95": round(_percentile(latencies, 95), 1),
                "p99": round(_percentile(latencies, 99), 1),
                "sample_count": len(latencies),
            },
            "ttft_ms": {
                "p50": round(_percentile(ttft_samples, 50), 1),
                "p95": round(_percentile(ttft_samples, 95), 1),
                "p99": round(_percentile(ttft_samples, 99), 1),
                "sample_count": len(ttft_samples),
            },
            "itl_ms": {
                "p50": round(_percentile(itl_samples, 50), 1),
                "p95": round(_percentile(itl_samples, 95), 1),
                "sample_count": len(itl_samples),
            },
            "queue_wait_ms": {
                "p50": round(_percentile(queue_waits, 50), 1),
                "p95": round(_percentile(queue_waits, 95), 1),
                "sample_count": len(queue_waits),
            },
            "gpu_utilization_avg_pct": round(avg_gpu_utilization, 1),
        }

    def get_tenant_summary(self, tenant_id: uuid.UUID) -> dict[str, Any]:
        """Get request counters for a specific tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Dict with per-status request counts and error rate.
        """
        counters = dict(self._tenant_counters.get(tenant_id, {}))
        total = sum(counters.values())
        errors = counters.get("error", 0) + counters.get("rate_limited", 0)

        return {
            "tenant_id": str(tenant_id),
            "total_requests": total,
            "counters": counters,
            "error_rate_pct": round(errors / max(total, 1) * 100, 2),
        }

    def get_error_rate(self, model: str, window_size: int = 100) -> float:
        """Get the recent error rate for a model.

        Args:
            model: Model identifier.
            window_size: Number of recent requests to consider.

        Returns:
            Error rate as a float between 0.0 and 1.0.
        """
        if model not in self._model_metrics:
            return 0.0
        metrics = self._model_metrics[model]
        total = min(metrics.total_requests, window_size)
        errors = min(metrics.error_requests, total)
        return errors / max(total, 1)

    def get_throughput(self, model: str) -> float:
        """Get the current EMA throughput for a model in tokens per second.

        Args:
            model: Model identifier.

        Returns:
            Exponential moving average of tokens/sec, or 0.0 if no data.
        """
        return self._model_metrics.get(model, ModelMetrics(model=model)).throughput_tokens_per_sec

    def list_active_models(self) -> list[str]:
        """List all models with recorded metrics.

        Returns:
            List of model identifiers that have at least one recorded request.
        """
        return [m for m, metrics in self._model_metrics.items() if metrics.total_requests > 0]

    def get_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus exposition format.

        Returns:
            String in Prometheus text format suitable for a /metrics endpoint.
        """
        lines: list[str] = []
        uptime_seconds = time.monotonic() - self._started_at

        lines.append("# HELP aumos_llm_requests_total Total LLM requests by model and status")
        lines.append("# TYPE aumos_llm_requests_total counter")

        lines.append("# HELP aumos_llm_tokens_total Total tokens processed by model and direction")
        lines.append("# TYPE aumos_llm_tokens_total counter")

        lines.append("# HELP aumos_llm_throughput_tokens_per_second Current token throughput EMA")
        lines.append("# TYPE aumos_llm_throughput_tokens_per_second gauge")

        lines.append("# HELP aumos_llm_latency_ms End-to-end request latency percentiles in ms")
        lines.append("# TYPE aumos_llm_latency_ms gauge")

        lines.append("# HELP aumos_llm_ttft_ms Time-to-first-token percentiles in ms")
        lines.append("# TYPE aumos_llm_ttft_ms gauge")

        lines.append("# HELP aumos_llm_error_rate_pct Error rate percentage by model")
        lines.append("# TYPE aumos_llm_error_rate_pct gauge")

        lines.append("# HELP aumos_llm_gpu_utilization_pct GPU utilization percentage by model")
        lines.append("# TYPE aumos_llm_gpu_utilization_pct gauge")

        lines.append("# HELP aumos_llm_uptime_seconds Service uptime in seconds")
        lines.append("# TYPE aumos_llm_uptime_seconds counter")
        lines.append(f'aumos_llm_uptime_seconds {uptime_seconds:.1f}')

        for model, metrics in self._model_metrics.items():
            safe_model = model.replace("/", "_").replace("-", "_").replace(".", "_")
            label = f'model="{model}"'

            # Request counters
            lines.append(f'aumos_llm_requests_total{{{label},status="success"}} {metrics.successful_requests}')
            lines.append(f'aumos_llm_requests_total{{{label},status="error"}} {metrics.error_requests}')
            lines.append(f'aumos_llm_requests_total{{{label},status="total"}} {metrics.total_requests}')

            # Token counters
            lines.append(f'aumos_llm_tokens_total{{{label},direction="prompt"}} {metrics.total_prompt_tokens}')
            lines.append(f'aumos_llm_tokens_total{{{label},direction="completion"}} {metrics.total_completion_tokens}')

            # Throughput
            lines.append(f'aumos_llm_throughput_tokens_per_second{{{label}}} {metrics.throughput_tokens_per_sec:.1f}')

            # Latency percentiles
            latencies = sorted(metrics.latency_samples_ms)
            for pct in (50, 95, 99):
                value = _percentile(latencies, pct)
                lines.append(f'aumos_llm_latency_ms{{{label},quantile="{pct}"}} {value:.1f}')

            # TTFT percentiles
            ttft_samples = sorted(metrics.ttft_samples_ms)
            for pct in (50, 95, 99):
                value = _percentile(ttft_samples, pct)
                lines.append(f'aumos_llm_ttft_ms{{{label},quantile="{pct}"}} {value:.1f}')

            # Error rate
            error_rate = metrics.error_requests / max(metrics.total_requests, 1) * 100
            lines.append(f'aumos_llm_error_rate_pct{{{label}}} {error_rate:.2f}')

            # GPU utilization
            gpu_samples = list(self._gpu_utilization.get(model, []))
            if gpu_samples:
                avg_gpu = statistics.mean(s[1] for s in gpu_samples[-100:])
                lines.append(f'aumos_llm_gpu_utilization_pct{{{label}}} {avg_gpu:.1f}')

        return "\n".join(lines) + "\n"

    def reset_model_metrics(self, model: str) -> None:
        """Clear all recorded metrics for a model (for testing/admin use).

        Args:
            model: Model identifier to reset.
        """
        if model in self._model_metrics:
            self._model_metrics[model] = ModelMetrics(model=model)
            logger.info("Model metrics reset", model=model)
