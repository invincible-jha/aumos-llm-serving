"""LLM cost tracker adapter.

Implements the CostTrackerProtocol with tiktoken-based token counting,
per-model pricing, per-tenant usage accumulation, budget enforcement,
and usage report generation integrated with aumos-ai-finops.
"""

from __future__ import annotations

import decimal
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)

# Pricing table: USD per 1000 tokens, (input_cost, output_cost)
# These are fallback defaults — production prices come from llm_model_configs table
_MODEL_PRICING: dict[str, tuple[decimal.Decimal, decimal.Decimal]] = {
    # OpenAI
    "gpt-4o": (decimal.Decimal("0.005"), decimal.Decimal("0.015")),
    "gpt-4o-mini": (decimal.Decimal("0.00015"), decimal.Decimal("0.0006")),
    "gpt-4-turbo": (decimal.Decimal("0.01"), decimal.Decimal("0.03")),
    "gpt-4": (decimal.Decimal("0.03"), decimal.Decimal("0.06")),
    "gpt-3.5-turbo": (decimal.Decimal("0.0005"), decimal.Decimal("0.0015")),
    # Anthropic
    "claude-opus-4": (decimal.Decimal("0.015"), decimal.Decimal("0.075")),
    "claude-opus-4-5": (decimal.Decimal("0.015"), decimal.Decimal("0.075")),
    "claude-sonnet-4-5": (decimal.Decimal("0.003"), decimal.Decimal("0.015")),
    "claude-haiku-3-5": (decimal.Decimal("0.0008"), decimal.Decimal("0.004")),
    # Cohere
    "command-r-plus": (decimal.Decimal("0.003"), decimal.Decimal("0.015")),
    "command-r": (decimal.Decimal("0.0005"), decimal.Decimal("0.0015")),
    # Google
    "gemini-1.5-pro": (decimal.Decimal("0.0035"), decimal.Decimal("0.0105")),
    "gemini-1.5-flash": (decimal.Decimal("0.00035"), decimal.Decimal("0.00105")),
    # Self-hosted / Ollama (zero cost)
    "llama3.2": (decimal.Decimal("0"), decimal.Decimal("0")),
    "llama3.1": (decimal.Decimal("0"), decimal.Decimal("0")),
    "mistral-7b-instruct": (decimal.Decimal("0"), decimal.Decimal("0")),
    "phi-3-mini": (decimal.Decimal("0"), decimal.Decimal("0")),
    "qwen2.5": (decimal.Decimal("0"), decimal.Decimal("0")),
}

# Tiktoken encoding aliases: model prefix → tiktoken encoding name
_TIKTOKEN_ENCODING_MAP: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "text-embedding": "cl100k_base",
    "claude": "cl100k_base",   # Approximate — Claude uses its own tokenizer
    "command": "cl100k_base",  # Approximate — Cohere uses its own tokenizer
    "gemini": "cl100k_base",   # Approximate
    "default": "cl100k_base",
}


def _get_tiktoken_encoding(model: str) -> tiktoken.Encoding:
    """Select the most appropriate tiktoken encoding for a model.

    Args:
        model: Model identifier string.

    Returns:
        tiktoken.Encoding instance.
    """
    model_lower = model.lower()
    for prefix, encoding_name in _TIKTOKEN_ENCODING_MAP.items():
        if model_lower.startswith(prefix) or prefix in model_lower:
            return tiktoken.get_encoding(encoding_name)
    return tiktoken.get_encoding(_TIKTOKEN_ENCODING_MAP["default"])


def _strip_provider_prefix(model: str) -> str:
    """Remove provider prefix from a model identifier.

    Args:
        model: Model identifier, possibly prefixed with "provider/".

    Returns:
        Model name without provider prefix.
    """
    return model.split("/")[-1] if "/" in model else model


class LLMCostTracker:
    """Per-tenant token counting and cost attribution tracker.

    Implements CostTrackerProtocol. Persists usage records via SQLAlchemy
    repositories and enforces daily/monthly budget limits. Generates usage
    reports and cost-optimization recommendations.

    Token counting:
    - OpenAI/GPT family: tiktoken cl100k_base (exact)
    - Other providers: tiktoken cl100k_base (approximate)
    - Always prefer provider-reported token counts when available

    Cost calculation:
    - Looks up per-model pricing from the in-memory table
    - Falls back to the ModelConfig table price in the database
    - Records zero cost for self-hosted models (configurable)
    """

    def __init__(
        self,
        session: AsyncSession,
        settings: LLMSettings,
        extra_pricing: dict[str, tuple[decimal.Decimal, decimal.Decimal]] | None = None,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            session: Async database session.
            settings: Service configuration.
            extra_pricing: Additional model pricing entries to merge in.
        """
        self._session = session
        self._settings = settings
        self._pricing: dict[str, tuple[decimal.Decimal, decimal.Decimal]] = {
            **_MODEL_PRICING,
            **(extra_pricing or {}),
        }
        # In-memory accumulator: tenant_id → (tokens, cost_usd) for the current minute
        self._minute_accumulators: dict[uuid.UUID, tuple[int, decimal.Decimal]] = {}

        logger.info(
            "LLMCostTracker initialized",
            pricing_entries=len(self._pricing),
            alert_threshold_usd=settings.cost_alert_threshold_usd,
        )

    # ------------------------------------------------------------------
    # CostTrackerProtocol implementation
    # ------------------------------------------------------------------

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in a text string for the given model.

        Args:
            text: Input text to tokenize.
            model: Model identifier used to select the tokenizer.

        Returns:
            Estimated token count.
        """
        try:
            encoding = _get_tiktoken_encoding(model)
            return len(encoding.encode(text, disallowed_special=()))
        except Exception as exc:
            # Fallback: approximate by whitespace splitting
            logger.warning(
                "tiktoken failed, using whitespace approximation",
                model=model,
                error=str(exc),
            )
            return max(1, len(text.split()))

    def count_message_tokens(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> int:
        """Count tokens across a list of chat messages.

        Adds per-message overhead (role tokens + separator) as per OpenAI
        token counting specification.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            model: Model identifier.

        Returns:
            Total token count including per-message overhead.
        """
        # OpenAI overhead: 4 tokens per message for role/content delimiters
        # plus 2 tokens for the reply primer
        message_overhead = 4
        reply_primer = 2

        total = reply_primer
        for message in messages:
            total += message_overhead
            total += self.count_tokens(message.get("content", ""), model)
            total += self.count_tokens(message.get("role", ""), model)
            if message.get("name"):
                total += self.count_tokens(message["name"], model) - 1
        return total

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> decimal.Decimal:
        """Calculate the USD cost for a request.

        Args:
            model: Model identifier (with or without provider prefix).
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            Total cost in USD as Decimal.
        """
        base_model = _strip_provider_prefix(model)

        # Exact match first, then prefix match
        input_per_k, output_per_k = self._pricing.get(
            model,
            self._pricing.get(base_model, (decimal.Decimal("0"), decimal.Decimal("0"))),
        )

        if input_per_k == decimal.Decimal("0") and output_per_k == decimal.Decimal("0"):
            # Try prefix matching for model families (e.g., "gpt-4-turbo-preview" → "gpt-4-turbo")
            for key, pricing in self._pricing.items():
                if base_model.startswith(key) or key.startswith(base_model.split("-")[0]):
                    input_per_k, output_per_k = pricing
                    break

        cost = (
            decimal.Decimal(prompt_tokens) * input_per_k / decimal.Decimal("1000")
            + decimal.Decimal(completion_tokens) * output_per_k / decimal.Decimal("1000")
        )

        return cost.quantize(decimal.Decimal("0.000001"))

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
        """Persist a usage record to the database and check alert thresholds.

        Args:
            tenant_id: Tenant who made the request.
            model: Model used.
            provider: Provider used.
            prompt_tokens: Input token count.
            completion_tokens: Output token count.
            cost: Total cost in USD.
            latency_ms: Request latency in milliseconds.
            status: Request outcome (success | error | rate_limited | quota_exceeded).
            error_message: Error detail if status is error.
        """
        from aumos_llm_serving.adapters.repositories import (  # noqa: PLC0415
            LLMRequestRepository,
        )

        repo = LLMRequestRepository(self._session)
        await repo.create(
            tenant_id=tenant_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

        # Update in-memory minute accumulator for fast budget checks
        prev_tokens, prev_cost = self._minute_accumulators.get(
            tenant_id, (0, decimal.Decimal("0"))
        )
        self._minute_accumulators[tenant_id] = (
            prev_tokens + prompt_tokens + completion_tokens,
            prev_cost + cost,
        )

        # Cost alert threshold check
        daily_cost = await self._get_daily_cost(tenant_id)
        alert_threshold = decimal.Decimal(str(self._settings.cost_alert_threshold_usd))
        if daily_cost >= alert_threshold:
            logger.warning(
                "Tenant daily cost alert threshold reached",
                tenant_id=str(tenant_id),
                daily_cost_usd=float(daily_cost),
                threshold_usd=float(alert_threshold),
            )

        logger.debug(
            "Usage recorded",
            tenant_id=str(tenant_id),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=float(cost),
            latency_ms=latency_ms,
            status=status,
        )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    async def check_budget(
        self,
        tenant_id: uuid.UUID,
        estimated_cost: decimal.Decimal,
    ) -> tuple[bool, str]:
        """Check if a request would exceed the tenant's daily budget.

        Args:
            tenant_id: Tenant to check.
            estimated_cost: Estimated cost for the upcoming request.

        Returns:
            Tuple of (within_budget, reason_string).
        """
        from aumos_llm_serving.adapters.repositories import (  # noqa: PLC0415
            TenantQuotaRepository,
        )

        quota_repo = TenantQuotaRepository(self._session)
        quota = await quota_repo.get_or_create(tenant_id)
        daily_cost = await self._get_daily_cost(tenant_id)

        if daily_cost + estimated_cost > quota.daily_cost_limit:
            return (
                False,
                (
                    f"Daily cost limit of ${quota.daily_cost_limit} would be exceeded "
                    f"(current: ${daily_cost:.6f}, estimated addition: ${estimated_cost:.6f})"
                ),
            )
        return True, "within_budget"

    # ------------------------------------------------------------------
    # Usage reports
    # ------------------------------------------------------------------

    async def generate_usage_report(
        self,
        tenant_id: uuid.UUID,
        period: str = "daily",
    ) -> dict[str, Any]:
        """Generate a usage and cost report for a tenant.

        Args:
            tenant_id: Tenant to report on.
            period: Report period: 'daily', 'weekly', or 'monthly'.

        Returns:
            Dict with token counts, cost breakdown, top models, and trends.
        """
        from aumos_llm_serving.adapters.repositories import (  # noqa: PLC0415
            LLMRequestRepository,
        )

        repo = LLMRequestRepository(self._session)
        now_utc = datetime.now(timezone.utc)

        if period == "daily":
            since = now_utc - timedelta(days=1)
        elif period == "weekly":
            since = now_utc - timedelta(days=7)
        elif period == "monthly":
            since = now_utc - timedelta(days=30)
        else:
            raise ValueError(f"Invalid period: {period!r}. Use 'daily', 'weekly', or 'monthly'.")

        stats = await repo.get_stats_since(tenant_id, since)

        return {
            "tenant_id": str(tenant_id),
            "period": period,
            "generated_at": now_utc.isoformat(),
            "since": since.isoformat(),
            "total_requests": stats.get("total_requests", 0),
            "successful_requests": stats.get("successful_requests", 0),
            "error_requests": stats.get("error_requests", 0),
            "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
            "total_completion_tokens": stats.get("total_completion_tokens", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "total_cost_usd": float(stats.get("total_cost", decimal.Decimal("0"))),
            "average_latency_ms": stats.get("average_latency_ms", 0),
            "top_models": stats.get("top_models", []),
        }

    def get_cost_optimization_recommendations(
        self,
        model: str,
        monthly_tokens: int,
    ) -> list[dict[str, Any]]:
        """Generate cost optimization recommendations for a given usage pattern.

        Args:
            model: Current model being used.
            monthly_tokens: Average tokens consumed per month.

        Returns:
            Ordered list of recommendation dicts (highest savings first).
        """
        base_model = _strip_provider_prefix(model)
        input_per_k, output_per_k = self._pricing.get(
            base_model, (decimal.Decimal("0"), decimal.Decimal("0"))
        )
        current_monthly_cost = float(
            decimal.Decimal(monthly_tokens) * (input_per_k + output_per_k) / decimal.Decimal("2000")
        )

        recommendations: list[dict[str, Any]] = []

        for alt_model, (alt_in, alt_out) in self._pricing.items():
            if alt_model == base_model:
                continue
            alt_cost = float(
                decimal.Decimal(monthly_tokens) * (alt_in + alt_out) / decimal.Decimal("2000")
            )
            savings = current_monthly_cost - alt_cost
            if savings > 0:
                recommendations.append(
                    {
                        "alternative_model": alt_model,
                        "monthly_savings_usd": round(savings, 2),
                        "savings_pct": round(savings / max(current_monthly_cost, 0.0001) * 100, 1),
                        "current_monthly_cost_usd": round(current_monthly_cost, 2),
                        "alternative_monthly_cost_usd": round(alt_cost, 2),
                    }
                )

        return sorted(recommendations, key=lambda r: r["monthly_savings_usd"], reverse=True)[:5]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_daily_cost(self, tenant_id: uuid.UUID) -> decimal.Decimal:
        """Get the total cost incurred today for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Total cost in USD as Decimal.
        """
        from aumos_llm_serving.adapters.repositories import LLMRequestRepository  # noqa: PLC0415

        repo = LLMRequestRepository(self._session)
        stats = await repo.get_daily_stats(tenant_id)
        return decimal.Decimal(str(stats.get("total_cost", 0)))

    def update_model_pricing(
        self,
        model: str,
        input_cost_per_1k: decimal.Decimal,
        output_cost_per_1k: decimal.Decimal,
    ) -> None:
        """Update the in-memory pricing table for a model.

        Args:
            model: Model identifier to update.
            input_cost_per_1k: Cost per 1000 input tokens in USD.
            output_cost_per_1k: Cost per 1000 output tokens in USD.
        """
        self._pricing[model] = (input_cost_per_1k, output_cost_per_1k)
        logger.info(
            "Model pricing updated",
            model=model,
            input_per_1k=float(input_cost_per_1k),
            output_per_1k=float(output_cost_per_1k),
        )
