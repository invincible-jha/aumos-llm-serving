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
