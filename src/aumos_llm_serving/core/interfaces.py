"""Protocol interfaces for the LLM serving service.

Defines abstract contracts that all provider adapters, routing components,
and tracking services must implement. Using Protocol (structural subtyping)
rather than ABC to keep adapters decoupled from the core package.
"""

from __future__ import annotations

import decimal
import uuid
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from aumos_llm_serving.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Contract for all LLM provider adapters.

    Every provider (vLLM, Ollama, LiteLLM, OpenAI) must implement this
    interface. The ServingService uses this to call providers without
    knowing their implementation details.
    """

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name (e.g., 'vllm', 'ollama')."""
        ...

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> ChatCompletionResponse:
        """Execute a chat completion request.

        Args:
            request: OpenAI-compatible chat completion request.
            model_override: If provided, use this model instead of request.model.

        Returns:
            OpenAI-compatible chat completion response.
        """
        ...

    async def text_completion(
        self,
        request: TextCompletionRequest,
        model_override: str | None = None,
    ) -> TextCompletionResponse:
        """Execute a text completion request.

        Args:
            request: OpenAI-compatible text completion request.
            model_override: If provided, use this model instead of request.model.

        Returns:
            OpenAI-compatible text completion response.
        """
        ...

    async def embed(
        self,
        request: EmbeddingRequest,
        model_override: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for input text.

        Args:
            request: OpenAI-compatible embedding request.
            model_override: If provided, use this model instead of request.model.

        Returns:
            OpenAI-compatible embedding response.
        """
        ...

    async def list_models(self) -> list[str]:
        """Return list of available model identifiers.

        Returns:
            List of model IDs available from this provider.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is healthy and reachable.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        ...

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response as server-sent events.

        Args:
            request: OpenAI-compatible chat completion request (stream=True).
            model_override: If provided, use this model instead of request.model.

        Yields:
            SSE data chunks in OpenAI streaming format.
        """
        ...


@runtime_checkable
class ModelRouterProtocol(Protocol):
    """Contract for intelligent model routing.

    Selects the best provider and model for a given request based on
    task type, cost constraints, latency requirements, and provider health.
    """

    async def route(
        self,
        request: ChatCompletionRequest | TextCompletionRequest | EmbeddingRequest,
        tenant_id: uuid.UUID,
    ) -> tuple[LLMProviderProtocol, str]:
        """Select the best provider and model for a request.

        Args:
            request: The incoming LLM request.
            tenant_id: Tenant context for quota and config lookup.

        Returns:
            Tuple of (provider_instance, model_name_to_use).
        """
        ...

    async def get_healthy_providers(self) -> list[str]:
        """Return names of currently healthy providers.

        Returns:
            List of healthy provider names.
        """
        ...


@runtime_checkable
class CostTrackerProtocol(Protocol):
    """Contract for token counting and cost tracking."""

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a given model.

        Args:
            text: The text to count tokens for.
            model: The model identifier (used to select tokenizer).

        Returns:
            Token count.
        """
        ...

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> decimal.Decimal:
        """Calculate the cost in USD for a request.

        Args:
            model: The model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Total cost in USD.
        """
        ...

    async def record_usage(
        self,
        tenant_id: uuid.UUID,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: decimal.Decimal,
        latency_ms: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Persist a usage record to the database.

        Args:
            tenant_id: Tenant who made the request.
            model: Model used.
            provider: Provider used.
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            cost: Total cost in USD.
            latency_ms: Request latency in milliseconds.
            status: Request status (success | error | rate_limited | quota_exceeded).
            error_message: Error details if status is error.
        """
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Contract for per-tenant rate limiting."""

    async def check_and_increment(
        self,
        tenant_id: uuid.UUID,
        tokens_requested: int,
        rpm_limit: int,
        tpm_limit: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Check rate limits and increment counters if allowed.

        Args:
            tenant_id: Tenant to check limits for.
            tokens_requested: Estimated tokens for this request.
            rpm_limit: Requests-per-minute limit for this tenant.
            tpm_limit: Tokens-per-minute limit for this tenant.

        Returns:
            Tuple of (is_allowed, rate_limit_headers) where headers contain
            X-RateLimit-* values for the response.
        """
        ...

    async def get_current_usage(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, int]:
        """Get current rate limit counters for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Dict with keys: requests_this_minute, tokens_this_minute.
        """
        ...


@runtime_checkable
class TokenCounterProtocol(Protocol):
    """Contract for token counting implementations."""

    def count(self, text: str, model: str) -> int:
        """Count tokens in text for a model.

        Args:
            text: Text to tokenize.
            model: Model identifier to select tokenizer.

        Returns:
            Token count.
        """
        ...

    def count_messages(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> int:
        """Count tokens across a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier to select tokenizer.

        Returns:
            Total token count across all messages.
        """
        ...


@runtime_checkable
class ModelLoaderProtocol(Protocol):
    """Contract for model discovery, loading, caching, and lifecycle management."""

    def discover_models(self) -> list[dict[str, Any]]:
        """Scan configured model root and return available models.

        Returns:
            List of dicts with model_id, format, path, and vram_estimate_bytes.
        """
        ...

    async def load_model(self, model_id: str) -> Any:
        """Load a model into memory with LRU cache eviction.

        Args:
            model_id: Model identifier to load.

        Returns:
            ModelEntry describing the loaded model.
        """
        ...

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model and reclaim VRAM.

        Args:
            model_id: Model to unload.

        Returns:
            True if unloaded, False if the model was not loaded.
        """
        ...

    def list_loaded_models(self) -> list[dict[str, Any]]:
        """List all currently loaded models.

        Returns:
            List of model status dicts.
        """
        ...

    def get_vram_usage(self) -> dict[str, int | float]:
        """Return current VRAM utilization.

        Returns:
            Dict with used_bytes, total_bytes, and utilization_pct.
        """
        ...


@runtime_checkable
class BatchSchedulerProtocol(Protocol):
    """Contract for dynamic LLM request batching."""

    async def start(self) -> None:
        """Start the background batching worker."""
        ...

    async def stop(self) -> None:
        """Stop the background worker and drain remaining requests."""
        ...

    async def submit(
        self,
        tenant_id: uuid.UUID,
        tenant_tier: Any,
        model: str,
        payload: dict[str, Any],
    ) -> Any:
        """Submit a request for batched inference.

        Args:
            tenant_id: Tenant submitting the request.
            tenant_tier: Scheduling priority tier.
            model: Model identifier.
            payload: Request payload dict.

        Returns:
            Inference result.
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return throughput and queue metrics.

        Returns:
            Dict with queue depths, throughput, latency stats, and counters.
        """
        ...


@runtime_checkable
class StreamHandlerProtocol(Protocol):
    """Contract for SSE token streaming management."""

    async def stream(
        self,
        source: AsyncIterator[str],
        tenant_id: uuid.UUID,
        model: str,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream SSE token chunks with timeout and heartbeat support.

        Args:
            source: Raw provider SSE chunk iterator.
            tenant_id: Owning tenant.
            model: Model producing the stream.
            timeout_seconds: Per-stream timeout override.

        Yields:
            Formatted SSE data chunks.
        """
        ...

    async def cancel_stream(self, stream_id: uuid.UUID) -> bool:
        """Cancel an active SSE stream.

        Args:
            stream_id: Stream to cancel.

        Returns:
            True if cancelled, False if not found.
        """
        ...

    def list_active_streams(self) -> list[dict[str, object]]:
        """List all active SSE streams.

        Returns:
            List of stream progress dicts.
        """
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Contract for inference performance metrics collection."""

    def record_request(self, trace: Any) -> None:
        """Record a completed request trace.

        Args:
            trace: Completed RequestTrace (from adapters.metrics_collector)
                   with all timing fields populated.
        """
        ...

    def record_gpu_utilization(self, model: str, utilization_pct: float) -> None:
        """Record a GPU utilization sample.

        Args:
            model: Model currently occupying the GPU.
            utilization_pct: GPU utilization percentage (0–100).
        """
        ...

    def get_model_summary(self, model: str) -> dict[str, Any]:
        """Get a performance summary for a model.

        Args:
            model: Model identifier.

        Returns:
            Dict with throughput, latency percentiles, TTFT, error rate, etc.
        """
        ...

    def get_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus exposition format.

        Returns:
            String in Prometheus text format for /metrics endpoint.
        """
        ...
