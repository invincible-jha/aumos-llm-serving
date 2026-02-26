"""Ollama provider adapter.

Integrates with Ollama for local development and small-scale deployments.
Ollama provides an OpenAI-compatible API at /api/*, with its own format
for model management.

Use cases:
- Local development without cloud API costs
- Air-gapped / on-premises deployments
- Small models (llama3.2, mistral, codellama, etc.)
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI

from aumos_common.observability import get_logger

from aumos_llm_serving.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    TextCompletionRequest,
    TextCompletionResponse,
    UsageInfo,
)
from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)


class OllamaProvider:
    """Ollama HTTP provider adapter.

    Uses Ollama's OpenAI-compatible /v1/* endpoint for chat and completions,
    and Ollama's native /api/embeddings for embedding generation.
    """

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize the Ollama provider.

        Args:
            settings: Service configuration with Ollama base URL.
        """
        self._settings = settings
        self._base_url = settings.ollama_base_url

        # Ollama exposes OpenAI-compatible endpoints at /v1
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",  # Ollama does not require a real API key
            timeout=httpx.Timeout(settings.provider_timeout_seconds),
            max_retries=settings.provider_max_retries,
        )
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(settings.provider_timeout_seconds),
        )

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        return "ollama"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> ChatCompletionResponse:
        """Execute a chat completion via Ollama.

        Args:
            request: OpenAI-compatible chat completion request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible chat completion response.
        """
        model = model_override or request.model

        messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in request.messages
        ]

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
            max_tokens=request.max_tokens,
        )

        logger.debug(
            "Ollama chat completion",
            model=model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        )

        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                {
                    "index": choice.index,
                    "message": ChatMessage(
                        role=choice.message.role,  # type: ignore[arg-type]
                        content=choice.message.content,
                    ),
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    async def text_completion(
        self,
        request: TextCompletionRequest,
        model_override: str | None = None,
    ) -> TextCompletionResponse:
        """Execute a text completion via Ollama.

        Args:
            request: OpenAI-compatible text completion request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible text completion response.
        """
        model = model_override or request.model

        response = await self._client.completions.create(
            model=model,
            prompt=request.prompt,  # type: ignore[arg-type]
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )

        return TextCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                {
                    "text": choice.text,
                    "index": choice.index,
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    async def embed(
        self,
        request: EmbeddingRequest,
        model_override: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings via Ollama's native /api/embed endpoint.

        Args:
            request: OpenAI-compatible embedding request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible embedding response.
        """
        model = model_override or request.model

        # Normalise input to list of strings
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list) and all(isinstance(i, str) for i in request.input):
            inputs = list(request.input)  # type: ignore[arg-type]
        else:
            inputs = [str(request.input)]

        # Ollama /api/embed accepts a list
        response = await self._http.post(
            "/api/embed",
            json={"model": model, "input": inputs},
        )
        response.raise_for_status()
        data = response.json()

        embeddings: list[EmbeddingObject] = []
        for idx, emb in enumerate(data.get("embeddings", [])):
            embeddings.append(EmbeddingObject(embedding=emb, index=idx))

        prompt_tokens = data.get("prompt_eval_count", len(inputs))

        return EmbeddingResponse(
            data=embeddings,
            model=model,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    async def list_models(self) -> list[str]:
        """List models available from Ollama.

        Returns:
            List of model tags pulled in Ollama.
        """
        try:
            response = await self._http.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Failed to list Ollama models", error=str(exc))
            return []

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if the Ollama API responds to a version request.
        """
        try:
            response = await self._http.get("/api/version", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Ollama health check failed", error=str(exc))
            return False

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion from Ollama.

        Args:
            request: OpenAI-compatible chat completion request.
            model_override: Optional model name override.

        Yields:
            SSE data chunks in OpenAI streaming format.
        """
        model = model_override or request.model
        messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in request.messages
        ]

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=request.temperature,
            stream=True,
        )

        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
