"""LiteLLM provider adapter.

Provides unified access to 100+ LLM providers (OpenAI, Anthropic, Azure,
Cohere, AWS Bedrock, Google Gemini, etc.) through the LiteLLM SDK.
Handles provider routing, model aliases, fallback chains, cost calculation,
and error normalization across all providers.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator

import litellm
from litellm import acompletion, aembedding
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.observability import get_logger

from aumos_llm_serving.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingObject,
    TextCompletionRequest,
    TextCompletionResponse,
    UsageInfo,
)
from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)

# Cost table (USD per 1000 tokens): {model: (input_cost, output_cost)}
# Updated periodically — also populated from LiteLLM's own model_cost dict
_DEFAULT_COST_TABLE: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("0.005"), Decimal("0.015")),
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "gpt-4-turbo": (Decimal("0.01"), Decimal("0.03")),
    "gpt-3.5-turbo": (Decimal("0.0005"), Decimal("0.0015")),
    "claude-opus-4": (Decimal("0.015"), Decimal("0.075")),
    "claude-opus-4-5": (Decimal("0.015"), Decimal("0.075")),
    "claude-sonnet-4-5": (Decimal("0.003"), Decimal("0.015")),
    "claude-haiku-3-5": (Decimal("0.0008"), Decimal("0.004")),
    "command-r-plus": (Decimal("0.003"), Decimal("0.015")),
    "command-r": (Decimal("0.0005"), Decimal("0.0015")),
    "gemini-1.5-pro": (Decimal("0.0035"), Decimal("0.0105")),
    "gemini-1.5-flash": (Decimal("0.00035"), Decimal("0.00105")),
    "llama3.2": (Decimal("0.0"), Decimal("0.0")),
    "mistral-7b-instruct": (Decimal("0.0002"), Decimal("0.0002")),
}

# Model aliases: maps short alias → canonical LiteLLM model ID
_MODEL_ALIASES: dict[str, str] = {
    "gpt-4": "gpt-4o",
    "gpt-4-mini": "gpt-4o-mini",
    "claude-3": "claude-opus-4",
    "claude-3-sonnet": "claude-sonnet-4-5",
    "claude-3-haiku": "claude-haiku-3-5",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    "cohere": "command-r-plus",
}


def _normalize_provider_error(exc: Exception, provider: str, model: str) -> Exception:
    """Convert provider-specific exceptions to AumOS standard errors.

    Args:
        exc: The original exception from LiteLLM.
        provider: Provider name for logging context.
        model: Model identifier for logging context.

    Returns:
        An AumOS standard error (ValidationError or NotFoundError).
    """
    if isinstance(exc, AuthenticationError):
        logger.error(
            "LiteLLM authentication failed",
            provider=provider,
            model=model,
        )
        return ValidationError(
            message=f"Authentication failed for provider: {provider}",
            field="api_key",
            value="[REDACTED]",
        )
    if isinstance(exc, RateLimitError):
        return ValidationError(
            message=f"Provider rate limit exceeded: {provider}",
            field="rate_limit",
            value=model,
        )
    if isinstance(exc, BadRequestError):
        return ValidationError(
            message=f"Invalid request to provider {provider}: {exc}",
            field="request",
            value=model,
        )
    if isinstance(exc, (ServiceUnavailableError, Timeout)):
        return NotFoundError(resource_type="LLMProvider", resource_id=provider)
    return exc


class LiteLLMProvider:
    """Unified LLM provider via LiteLLM SDK.

    Wraps LiteLLM to provide access to 100+ cloud LLM providers through
    a single interface. Supports provider routing, model aliases, fallback
    chains, token counting, and cost calculation.

    Provider selection is driven by the model string prefix:
    - "openai/gpt-4o" → OpenAI
    - "anthropic/claude-opus-4" → Anthropic
    - "azure/gpt-4" → Azure OpenAI
    - "cohere/command-r-plus" → Cohere
    - "bedrock/llama3-70b" → AWS Bedrock
    - "gemini/gemini-1.5-pro" → Google Gemini
    """

    def __init__(
        self,
        settings: LLMSettings,
        fallback_models: list[str] | None = None,
        extra_model_aliases: dict[str, str] | None = None,
    ) -> None:
        """Initialize the LiteLLM provider.

        Args:
            settings: Service configuration with provider keys and timeouts.
            fallback_models: Ordered list of fallback model IDs if primary fails.
            extra_model_aliases: Additional model aliases to register.
        """
        self._settings = settings
        self._fallback_models: list[str] = fallback_models or []
        self._model_aliases: dict[str, str] = {**_MODEL_ALIASES, **(extra_model_aliases or {})}
        self._cost_table: dict[str, tuple[Decimal, Decimal]] = dict(_DEFAULT_COST_TABLE)

        # Configure LiteLLM globally
        litellm.set_verbose = False
        litellm.request_timeout = int(settings.provider_timeout_seconds)
        litellm.num_retries = settings.provider_max_retries

        # Populate cost table from LiteLLM's built-in pricing
        self._sync_litellm_cost_table()

        logger.info(
            "LiteLLM provider initialized",
            fallback_models=self._fallback_models,
            alias_count=len(self._model_aliases),
        )

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        return "litellm"

    def resolve_model_alias(self, model: str) -> str:
        """Resolve a model alias to its canonical LiteLLM model ID.

        Args:
            model: Model identifier, possibly an alias.

        Returns:
            Canonical LiteLLM model ID.
        """
        return self._model_aliases.get(model, model)

    def _sync_litellm_cost_table(self) -> None:
        """Populate cost table from LiteLLM's built-in model pricing."""
        try:
            litellm_costs: dict[str, Any] = getattr(litellm, "model_cost", {})
            for model_id, cost_info in litellm_costs.items():
                if isinstance(cost_info, dict):
                    input_cost = Decimal(str(cost_info.get("input_cost_per_token", 0))) * 1000
                    output_cost = Decimal(str(cost_info.get("output_cost_per_token", 0))) * 1000
                    # Only override if not in our curated table
                    if model_id not in self._cost_table:
                        self._cost_table[model_id] = (input_cost, output_cost)
        except Exception as exc:
            logger.warning("Failed to sync LiteLLM cost table", error=str(exc))

    def calculate_request_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Decimal:
        """Calculate the cost in USD for a LiteLLM request.

        Args:
            model: Model identifier (canonical or alias).
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            Total cost in USD as Decimal.
        """
        canonical_model = self.resolve_model_alias(model)
        # Strip provider prefix for cost table lookup
        base_model = canonical_model.split("/")[-1] if "/" in canonical_model else canonical_model

        input_cost_per_k, output_cost_per_k = self._cost_table.get(
            canonical_model,
            self._cost_table.get(base_model, (Decimal("0"), Decimal("0"))),
        )

        cost = (
            Decimal(prompt_tokens) * input_cost_per_k / Decimal("1000")
            + Decimal(completion_tokens) * output_cost_per_k / Decimal("1000")
        )
        return cost

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> ChatCompletionResponse:
        """Execute a chat completion via LiteLLM routing.

        Attempts the primary model first, then falls back through the
        configured fallback chain on failures.

        Args:
            request: OpenAI-compatible chat completion request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            ValidationError: On authentication or bad-request errors.
            NotFoundError: When no provider in the chain is available.
        """
        raw_model = model_override or request.model
        model = self.resolve_model_alias(raw_model)

        messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in request.messages
        ]

        # Build fallback list: primary + configured fallbacks
        fallbacks = [model] + self._fallback_models

        last_exc: Exception | None = None
        for candidate_model in fallbacks:
            try:
                start_time = time.monotonic()
                response = await acompletion(
                    model=candidate_model,
                    messages=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    n=request.n or 1,
                    stream=False,
                    stop=request.stop,
                    max_tokens=request.max_tokens,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    seed=request.seed,
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                usage = response.usage
                logger.info(
                    "LiteLLM chat completion",
                    model=candidate_model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=elapsed_ms,
                )

                return ChatCompletionResponse(
                    id=response.id or f"litellm-{uuid.uuid4()}",
                    created=int(response.created or time.time()),
                    model=response.model or candidate_model,
                    choices=[
                        {
                            "index": choice.index,
                            "message": ChatMessage(
                                role=choice.message.role,
                                content=choice.message.content,
                            ),
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                    ],
                    usage=UsageInfo(
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        total_tokens=usage.total_tokens if usage else 0,
                    ),
                )

            except (AuthenticationError, BadRequestError) as exc:
                # Non-retriable: fail immediately
                raise _normalize_provider_error(exc, candidate_model, candidate_model) from exc

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LiteLLM model failed, trying fallback",
                    failed_model=candidate_model,
                    error=str(exc),
                )
                continue

        raise NotFoundError(resource_type="LLMProvider", resource_id=model) from last_exc

    async def text_completion(
        self,
        request: TextCompletionRequest,
        model_override: str | None = None,
    ) -> TextCompletionResponse:
        """Execute a text completion via LiteLLM.

        Args:
            request: OpenAI-compatible text completion request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible text completion response.
        """
        raw_model = model_override or request.model
        model = self.resolve_model_alias(raw_model)

        # LiteLLM routes text completions through chat for non-OpenAI providers
        prompt_text = (
            request.prompt
            if isinstance(request.prompt, str)
            else str(request.prompt)
        )
        messages = [{"role": "user", "content": prompt_text}]

        response = await acompletion(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n or 1,
            stop=request.stop,
        )

        usage = response.usage
        content = response.choices[0].message.content if response.choices else ""

        return TextCompletionResponse(
            id=response.id or f"litellm-{uuid.uuid4()}",
            created=int(response.created or time.time()),
            model=response.model or model,
            choices=[
                {
                    "text": content or "",
                    "index": 0,
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                }
            ],
            usage=UsageInfo(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def embed(
        self,
        request: EmbeddingRequest,
        model_override: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings via LiteLLM.

        Args:
            request: OpenAI-compatible embedding request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible embedding response.
        """
        raw_model = model_override or request.model
        model = self.resolve_model_alias(raw_model)

        response = await aembedding(model=model, input=request.input)

        usage = response.usage
        return EmbeddingResponse(
            data=[
                EmbeddingObject(
                    object="embedding",
                    embedding=item.embedding,
                    index=item.index,
                )
                for item in response.data
            ],
            model=response.model or model,
            usage=EmbeddingUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def list_models(self) -> list[str]:
        """Return the list of models accessible via LiteLLM.

        Returns:
            List of canonical LiteLLM model IDs (aliases included).
        """
        canonical_models = list(self._cost_table.keys())
        alias_models = list(self._model_aliases.keys())
        return sorted(set(canonical_models + alias_models))

    async def health_check(self) -> bool:
        """Check if LiteLLM is operational by listing supported models.

        Returns:
            True if LiteLLM SDK is functional.
        """
        try:
            _ = await self.list_models()
            return True
        except Exception as exc:
            logger.warning("LiteLLM health check failed", error=str(exc))
            return False

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens via LiteLLM SSE.

        Args:
            request: OpenAI-compatible chat completion request with stream=True.
            model_override: Optional model name override.

        Yields:
            SSE data chunks in OpenAI streaming format.
        """
        raw_model = model_override or request.model
        model = self.resolve_model_alias(raw_model)

        messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in request.messages
        ]

        stream = await acompletion(
            model=model,
            messages=messages,
            temperature=request.temperature,
            stream=True,
        )

        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
