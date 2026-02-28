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


# =============================================================================
# AumOS Extensions: Streaming Analytics (Gap #139)
# =============================================================================


class StreamingMetricsResponse(BaseModel):
    """Per-tenant streaming usage analytics."""

    tenant_id: uuid.UUID
    period_hours: int
    total_streaming_requests: int
    total_streaming_tokens: int
    avg_tokens_per_second: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float


# =============================================================================
# AumOS Extensions: A/B Model Testing (Gap #140)
# =============================================================================


class ABTestCreateRequest(BaseModel):
    """Request to create an A/B model test experiment."""

    name: str = Field(..., max_length=200, description="Experiment name")
    model_a: str = Field(..., description="First model identifier")
    model_b: str = Field(..., description="Second model identifier")
    traffic_split_pct: int = Field(
        default=50, ge=1, le=99, description="Percentage of traffic routed to model_a"
    )
    evaluation_metric: str = Field(
        default="latency_ms",
        description="Primary metric: latency_ms, cost_usd, quality_score",
    )
    sample_size: int = Field(default=1000, ge=10, le=100000, description="Target request count per arm")


class ABTestResponse(BaseModel):
    """Response for an A/B test experiment."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    model_a: str
    model_b: str
    traffic_split_pct: int
    evaluation_metric: str
    sample_size: int
    requests_a: int
    requests_b: int
    avg_metric_a: float | None
    avg_metric_b: float | None
    status: str
    winner: str | None
    created_at: Any

    model_config = {"from_attributes": True}


class ABTestListResponse(BaseModel):
    """List of A/B test experiments."""

    items: list[ABTestResponse]
    total: int


# =============================================================================
# AumOS Extensions: Content Guardrails (Gap #141)
# =============================================================================


class GuardrailRuleCreateRequest(BaseModel):
    """Request to create a content guardrail rule."""

    name: str = Field(..., max_length=200, description="Rule name")
    rule_type: str = Field(
        ...,
        description="Rule type: keyword_block, regex_block, topic_filter, pii_redact",
    )
    pattern: str = Field(..., description="Keyword, regex pattern, or topic name to match")
    action: str = Field(
        default="block",
        description="Action on match: block (reject request), redact (replace match), warn (log only)",
    )
    applies_to: str = Field(
        default="both",
        description="Where to apply: prompt, completion, both",
    )


class GuardrailRuleResponse(BaseModel):
    """Response for a guardrail rule."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    rule_type: str
    pattern: str
    action: str
    applies_to: str
    enabled: bool
    created_at: Any

    model_config = {"from_attributes": True}


# =============================================================================
# AumOS Extensions: Model Warm-Up (Gap #143)
# =============================================================================


class ModelWarmUpRequest(BaseModel):
    """Request to pre-warm a model on a specific provider."""

    model_name: str = Field(..., description="Model identifier to warm up")
    provider: str = Field(..., description="Provider to warm: vllm, ollama, litellm")
    sample_prompt: str = Field(
        default="Hello",
        max_length=500,
        description="Sample prompt to use for the warm-up inference",
    )


class ModelWarmUpResponse(BaseModel):
    """Result of a model warm-up request."""

    model_name: str
    provider: str
    latency_ms: int
    success: bool
    error: str | None = None


# =============================================================================
# AumOS Extensions: Cache Monitoring (Gap #144)
# =============================================================================


class CacheStatsResponse(BaseModel):
    """Response with KV cache statistics for model providers."""

    provider: str
    model_name: str | None
    cache_hit_rate: float
    cache_hits: int
    cache_misses: int
    evictions: int
    memory_used_gb: float
    memory_total_gb: float
    collected_at: Any


# =============================================================================
# AumOS Extensions: Admin Dashboard (Gap #137)
# =============================================================================


class AdminDashboardResponse(BaseModel):
    """Platform-wide admin dashboard data."""

    total_tenants: int
    active_tenants_today: int
    total_requests_today: int
    total_tokens_today: int
    total_cost_today_usd: float
    requests_by_model: dict[str, int]
    requests_by_provider: dict[str, int]
    error_rate_pct: float
    avg_latency_ms: float


# =============================================================================
# AumOS Extensions: Complexity Router Validation (Gap #138)
# =============================================================================


class RouterValidationRequest(BaseModel):
    """Request to validate routing decision for a given prompt."""

    prompt: str = Field(..., max_length=10000, description="Prompt to route")
    task_type: str | None = Field(default=None, description="Hint: code, embedding, fast, balanced")
    max_cost_usd: float | None = Field(default=None, ge=0, description="Cost constraint")
    max_latency_ms: int | None = Field(default=None, ge=1, description="Latency constraint in ms")


class RouterValidationResponse(BaseModel):
    """Result of routing validation showing which model would be selected."""

    selected_model: str
    selected_provider: str
    estimated_prompt_tokens: int
    estimated_cost_usd: float
    routing_reason: str
    alternative_models: list[dict[str, Any]]
