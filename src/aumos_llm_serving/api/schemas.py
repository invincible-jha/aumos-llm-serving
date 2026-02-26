"""Pydantic request/response schemas for the LLM serving API.

Two namespaces:
1. OpenAI-compatible schemas — mirrors the OpenAI API exactly for drop-in compatibility
2. AumOS extension schemas — for model management, quotas, and usage dashboards
"""

from __future__ import annotations

import decimal
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# OpenAI-Compatible: Chat Completions
# =============================================================================


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model identifier to use")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int | None = Field(default=None, ge=1, le=128)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, str] | None = None
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    """Single choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class UsageInfo(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: str | None = None


# =============================================================================
# OpenAI-Compatible: Text Completions
# =============================================================================


class TextCompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    suffix: str | None = None
    max_tokens: int | None = Field(default=16, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int | None = Field(default=None, ge=1, le=128)
    stream: bool = False
    logprobs: int | None = Field(default=None, ge=0, le=5)
    echo: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    best_of: int | None = Field(default=None, ge=1)
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class TextCompletionChoice(BaseModel):
    """Single choice in a text completion response."""

    text: str
    index: int
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None


class TextCompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextCompletionChoice]
    usage: UsageInfo


# =============================================================================
# OpenAI-Compatible: Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingObject(BaseModel):
    """Single embedding in a response."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


# =============================================================================
# OpenAI-Compatible: Models
# =============================================================================


class ModelObject(BaseModel):
    """OpenAI-compatible model info object."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: Literal["list"] = "list"
    data: list[dict[str, Any]]


# =============================================================================
# AumOS Extensions: Model Management
# =============================================================================


class ModelConfigCreateRequest(BaseModel):
    """Request to create a new model configuration."""

    model_name: str = Field(..., description="Model identifier (e.g., gpt-4, llama3.2)")
    provider: str = Field(..., description="Provider: vllm | ollama | openai | anthropic | azure | custom")
    endpoint_url: str | None = Field(default=None, description="Override endpoint URL")
    max_tokens: int = Field(default=4096, ge=1)
    temperature: decimal.Decimal = Field(default=decimal.Decimal("0.7"), ge=0, le=2)
    rate_limit_rpm: int | None = Field(default=None, ge=1)
    rate_limit_tpm: int | None = Field(default=None, ge=1)
    cost_per_input_token: decimal.Decimal | None = Field(default=None, ge=0)
    cost_per_output_token: decimal.Decimal | None = Field(default=None, ge=0)
    is_default: bool | None = None


class ModelConfigUpdateRequest(BaseModel):
    """Request to update an existing model configuration."""

    endpoint_url: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: decimal.Decimal | None = Field(default=None, ge=0, le=2)
    rate_limit_rpm: int | None = Field(default=None, ge=1)
    rate_limit_tpm: int | None = Field(default=None, ge=1)
    cost_per_input_token: decimal.Decimal | None = Field(default=None, ge=0)
    cost_per_output_token: decimal.Decimal | None = Field(default=None, ge=0)
    is_default: bool | None = None


class ModelConfigResponse(BaseModel):
    """Response containing a model configuration."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    model_name: str
    provider: str
    endpoint_url: str | None
    max_tokens: int
    temperature: decimal.Decimal
    rate_limit_rpm: int
    rate_limit_tpm: int
    cost_per_input_token: decimal.Decimal
    cost_per_output_token: decimal.Decimal
    is_default: bool

    model_config = {"from_attributes": True}


# =============================================================================
# AumOS Extensions: Tenant Quotas
# =============================================================================


class TenantQuotaRequest(BaseModel):
    """Request to set or update tenant quotas."""

    daily_token_limit: int = Field(..., ge=1, description="Maximum tokens per day")
    monthly_token_limit: int = Field(..., ge=1, description="Maximum tokens per month")
    daily_cost_limit: decimal.Decimal = Field(..., ge=0, description="Maximum USD spend per day")
    monthly_cost_limit: decimal.Decimal = Field(..., ge=0, description="Maximum USD spend per month")


class TenantQuotaResponse(BaseModel):
    """Response containing tenant quota configuration."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    daily_token_limit: int
    monthly_token_limit: int
    daily_tokens_used: int
    monthly_tokens_used: int
    daily_cost_limit: decimal.Decimal
    monthly_cost_limit: decimal.Decimal

    model_config = {"from_attributes": True}


# =============================================================================
# AumOS Extensions: Usage Dashboard
# =============================================================================


class TenantUsageResponse(BaseModel):
    """Tenant usage and cost dashboard response."""

    tenant_id: uuid.UUID
    daily_tokens_used: int
    daily_cost_usd: decimal.Decimal
    daily_token_limit: int
    daily_cost_limit: decimal.Decimal
    monthly_tokens_used: int
    monthly_cost_usd: decimal.Decimal
    monthly_token_limit: int
    monthly_cost_limit: decimal.Decimal

    @property
    def daily_token_utilization(self) -> float:
        """Percentage of daily token limit consumed."""
        if self.daily_token_limit == 0:
            return 0.0
        return self.daily_tokens_used / self.daily_token_limit * 100

    @property
    def monthly_token_utilization(self) -> float:
        """Percentage of monthly token limit consumed."""
        if self.monthly_token_limit == 0:
            return 0.0
        return self.monthly_tokens_used / self.monthly_token_limit * 100
