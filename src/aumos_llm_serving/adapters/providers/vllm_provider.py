"""vLLM provider adapter.

Integrates with vLLM's OpenAI-compatible HTTP server for production
GPU inference with PagedAttention and continuous batching.

vLLM exposes an OpenAI-compatible API at /v1/*, so this adapter
uses the OpenAI SDK pointed at the vLLM endpoint.
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
    EmbeddingRequest,
    EmbeddingResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    UsageInfo,
)
from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)


class VLLMProvider:
    """vLLM HTTP provider adapter.

    Communicates with a running vLLM server via its OpenAI-compatible API.
    Best for production deployments with GPU hardware.

    Features leveraged:
    - PagedAttention: efficient KV-cache management for long contexts
    - Continuous batching: serves multiple requests simultaneously
    - Token streaming: real-time response streaming
    - Tensor parallelism: multi-GPU model sharding (configured server-side)
    """

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize the vLLM provider.

        Args:
            settings: Service configuration with vLLM base URL and credentials.
        """
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.vllm_base_url + "/v1",
            api_key=settings.vllm_api_key or "none",
            timeout=httpx.Timeout(settings.provider_timeout_seconds),
            max_retries=settings.provider_max_retries,
        )

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        return "vllm"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> ChatCompletionResponse:
        """Execute a chat completion via vLLM.

        Args:
            request: OpenAI-compatible chat completion request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible chat completion response.
        """
        model = model_override or request.model

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
            if msg.content is not None
        ]

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n or 1,
            stream=False,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )

        logger.debug(
            "vLLM chat completion",
            model=model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
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
        """Execute a text completion via vLLM.

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
            top_p=request.top_p,
            n=request.n or 1,
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
        """Generate embeddings via vLLM.

        Args:
            request: OpenAI-compatible embedding request.
            model_override: Optional model name override.

        Returns:
            OpenAI-compatible embedding response.
        """
        model = model_override or request.model

        response = await self._client.embeddings.create(
            model=model,
            input=request.input,  # type: ignore[arg-type]
        )

        return EmbeddingResponse(
            data=[
                {
                    "object": "embedding",
                    "embedding": item.embedding,
                    "index": item.index,
                }
                for item in response.data
            ],
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    async def list_models(self) -> list[str]:
        """List models available from vLLM.

        Returns:
            List of model IDs loaded in the vLLM server.
        """
        response = await self._client.models.list()
        return [model.id for model in response.data]

    async def health_check(self) -> bool:
        """Check if vLLM server is reachable and healthy.

        Returns:
            True if the vLLM server responds to a model listing request.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._settings.vllm_base_url}/health")
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("vLLM health check failed", error=str(exc))
            return False

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion from vLLM.

        Args:
            request: OpenAI-compatible chat completion request with stream=True.
            model_override: Optional model name override.

        Yields:
            SSE data chunks in OpenAI streaming format.
        """
        model = model_override or request.model
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
            if msg.content is not None
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
