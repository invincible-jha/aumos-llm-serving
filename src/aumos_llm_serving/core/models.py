"""SQLAlchemy ORM models for the LLM serving service.

All tables use the `llm_` prefix and extend AumOSModel for automatic
tenant_id, id, created_at, updated_at fields with RLS support.
"""

import decimal

from sqlalchemy import DECIMAL, BigInteger, Boolean, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class LLMRequest(AumOSModel):
    """Records every LLM inference request for cost tracking and auditing.

    Each row represents one API call to a downstream LLM provider.
    Tenant isolation is enforced via RLS on tenant_id.
    """

    __tablename__ = "llm_requests"

    model: Mapped[str] = mapped_column(
        "model",
        nullable=False,
        comment="Model identifier used for this request (e.g., gpt-4, llama3.2)",
    )
    provider: Mapped[str] = mapped_column(
        "provider",
        nullable=False,
        comment="Provider that served this request: vllm | ollama | openai | anthropic | azure | custom",
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of tokens in the prompt/input",
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of tokens in the completion/output",
    )
    total_cost: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=12, scale=8),
        nullable=False,
        default=decimal.Decimal("0"),
        comment="Total cost in USD for this request",
    )
    latency_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="End-to-end latency in milliseconds (including provider round-trip)",
    )
    status: Mapped[str] = mapped_column(
        "status",
        nullable=False,
        default="success",
        comment="Request status: success | error | rate_limited | quota_exceeded",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Error message if status is error",
    )


class ModelConfig(AumOSModel):
    """Tenant-scoped model configuration.

    Defines which models are available to a tenant, with per-model routing
    parameters, cost rates, and rate limits.
    """

    __tablename__ = "llm_model_configs"

    model_name: Mapped[str] = mapped_column(
        "model_name",
        nullable=False,
        comment="Model identifier (e.g., gpt-4, llama3.2, claude-3-opus)",
    )
    provider: Mapped[str] = mapped_column(
        "provider",
        nullable=False,
        comment="Provider: vllm | ollama | openai | anthropic | azure | custom",
    )
    endpoint_url: Mapped[str | None] = mapped_column(
        "endpoint_url",
        nullable=True,
        comment="Override endpoint URL (for vLLM or custom deployments)",
    )
    max_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=4096,
        comment="Maximum output tokens allowed per request",
    )
    temperature: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=4, scale=3),
        nullable=False,
        default=decimal.Decimal("0.7"),
        comment="Default temperature for this model",
    )
    rate_limit_rpm: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=60,
        comment="Requests-per-minute limit for this model",
    )
    rate_limit_tpm: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100_000,
        comment="Tokens-per-minute limit for this model",
    )
    cost_per_input_token: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=12, scale=10),
        nullable=False,
        default=decimal.Decimal("0"),
        comment="Cost in USD per input token",
    )
    cost_per_output_token: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=12, scale=10),
        nullable=False,
        default=decimal.Decimal("0"),
        comment="Cost in USD per output token",
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this is the default model for this tenant",
    )


class TenantQuota(AumOSModel):
    """Per-tenant token and cost quotas.

    Controls daily and monthly consumption limits. The service enforces
    these limits before forwarding requests to providers.
    """

    __tablename__ = "llm_tenant_quotas"

    daily_token_limit: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1_000_000,
        comment="Maximum total tokens allowed per day",
    )
    monthly_token_limit: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=30_000_000,
        comment="Maximum total tokens allowed per month",
    )
    daily_tokens_used: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Tokens consumed today (reset daily via scheduled job)",
    )
    monthly_tokens_used: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Tokens consumed this month (reset monthly via scheduled job)",
    )
    daily_cost_limit: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=False,
        default=decimal.Decimal("50.0"),
        comment="Maximum spend in USD per day",
    )
    monthly_cost_limit: Mapped[decimal.Decimal] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=False,
        default=decimal.Decimal("500.0"),
        comment="Maximum spend in USD per month",
    )
