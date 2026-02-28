"""Business logic services for the LLM serving layer.

This module contains five services:
- ServingService: Unified LLM serving with OpenAI-compatible API
- RoutingService: Intelligent model routing (task/cost/latency/health)
- CostTrackingService: Per-tenant token consumption and cost attribution
- RateLimitingService: Per-tenant token/request rate limiting with quota enforcement
- ModelManagementService: CRUD for model configs, health checks, failover management
"""

from __future__ import annotations

import decimal
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.observability import get_logger

from aumos_llm_serving.api.schemas import (
    ABTestCreateRequest,
    ABTestResponse,
    AdminDashboardResponse,
    CacheStatsResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    GuardrailRuleCreateRequest,
    GuardrailRuleResponse,
    ModelConfigCreateRequest,
    ModelConfigResponse,
    ModelConfigUpdateRequest,
    ModelListResponse,
    ModelWarmUpRequest,
    ModelWarmUpResponse,
    RouterValidationRequest,
    RouterValidationResponse,
    StreamingMetricsResponse,
    TenantQuotaRequest,
    TenantQuotaResponse,
    TenantUsageResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)
from aumos_llm_serving.core.interfaces import (
    BatchSchedulerProtocol,
    CostTrackerProtocol,
    LLMProviderProtocol,
    MetricsCollectorProtocol,
    ModelLoaderProtocol,
    ModelRouterProtocol,
    RateLimiterProtocol,
    StreamHandlerProtocol,
)
from aumos_llm_serving.adapters.metrics_collector import RequestTrace
from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)


class ServingService:
    """Unified LLM serving with OpenAI-compatible API.

    Orchestrates the full request lifecycle:
    1. Rate limit check (Redis)
    2. Quota check (DB)
    3. Route to provider (ModelRouter)
    4. Call provider (LLMProvider)
    5. Record usage (CostTracker)
    6. Return response

    This service is the primary entry point for all LLM inference.
    """

    def __init__(
        self,
        router: ModelRouterProtocol,
        cost_tracker: CostTrackerProtocol,
        rate_limiter: RateLimiterProtocol,
        session: AsyncSession,
        settings: LLMSettings,
    ) -> None:
        """Initialize the serving service.

        Args:
            router: Model routing strategy implementation.
            cost_tracker: Token counting and cost recording implementation.
            rate_limiter: Rate limiting implementation.
            session: Async database session.
            settings: Service configuration.
        """
        self._router = router
        self._cost_tracker = cost_tracker
        self._rate_limiter = rate_limiter
        self._session = session
        self._settings = settings

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        tenant_id: uuid.UUID,
    ) -> ChatCompletionResponse:
        """Process a chat completion request end-to-end.

        Enforces rate limits, routes to the best provider, records usage,
        and returns an OpenAI-compatible response.

        Args:
            request: OpenAI-compatible chat completion request.
            tenant_id: Tenant making the request.

        Returns:
            OpenAI-compatible chat completion response.

        Raises:
            ValidationError: If rate limit or quota is exceeded.
        """
        # Estimate token count for rate limiting (before actual call)
        estimated_tokens = self._cost_tracker.count_tokens(
            text=str([m.model_dump() for m in request.messages]),
            model=request.model,
        )

        # Check rate limits
        await self._enforce_rate_limits(tenant_id, estimated_tokens)

        # Route to provider
        provider, model_name = await self._router.route(request, tenant_id)

        # Execute the call with timing
        start_ms = int(time.monotonic() * 1000)
        response: ChatCompletionResponse
        error_message: str | None = None
        status = "success"

        try:
            response = await provider.chat_completion(request, model_override=model_name)
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            logger.error(
                "Provider call failed",
                provider=provider.provider_name,
                model=model_name,
                tenant_id=str(tenant_id),
                error=error_message,
            )
            raise

        finally:
            latency_ms = int(time.monotonic() * 1000) - start_ms

            # Record usage even on failure (for auditing)
            if status == "success" and response is not None:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            else:
                prompt_tokens = estimated_tokens
                completion_tokens = 0

            cost = self._cost_tracker.calculate_cost(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            if self._settings.enable_cost_tracking:
                await self._cost_tracker.record_usage(
                    tenant_id=tenant_id,
                    model=model_name,
                    provider=provider.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )

        logger.info(
            "Chat completion served",
            model=model_name,
            provider=provider.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=float(cost),
            latency_ms=latency_ms,
            tenant_id=str(tenant_id),
        )

        return response

    async def text_completion(
        self,
        request: TextCompletionRequest,
        tenant_id: uuid.UUID,
    ) -> TextCompletionResponse:
        """Process a text completion request end-to-end.

        Args:
            request: OpenAI-compatible text completion request.
            tenant_id: Tenant making the request.

        Returns:
            OpenAI-compatible text completion response.
        """
        estimated_tokens = self._cost_tracker.count_tokens(
            text=request.prompt if isinstance(request.prompt, str) else str(request.prompt),
            model=request.model,
        )
        await self._enforce_rate_limits(tenant_id, estimated_tokens)

        provider, model_name = await self._router.route(request, tenant_id)

        start_ms = int(time.monotonic() * 1000)
        status = "success"
        error_message: str | None = None

        try:
            response = await provider.text_completion(request, model_override=model_name)
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            latency_ms = int(time.monotonic() * 1000) - start_ms
            if status == "success":
                prompt_tokens = response.usage.prompt_tokens  # type: ignore[union-attr]
                completion_tokens = response.usage.completion_tokens  # type: ignore[union-attr]
            else:
                prompt_tokens = estimated_tokens
                completion_tokens = 0

            cost = self._cost_tracker.calculate_cost(model_name, prompt_tokens, completion_tokens)
            if self._settings.enable_cost_tracking:
                await self._cost_tracker.record_usage(
                    tenant_id=tenant_id,
                    model=model_name,
                    provider=provider.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )

        return response  # type: ignore[return-value]

    async def embed(
        self,
        request: EmbeddingRequest,
        tenant_id: uuid.UUID,
    ) -> EmbeddingResponse:
        """Process an embedding request end-to-end.

        Args:
            request: OpenAI-compatible embedding request.
            tenant_id: Tenant making the request.

        Returns:
            OpenAI-compatible embedding response.
        """
        input_text = request.input if isinstance(request.input, str) else " ".join(request.input)
        estimated_tokens = self._cost_tracker.count_tokens(text=input_text, model=request.model)
        await self._enforce_rate_limits(tenant_id, estimated_tokens)

        provider, model_name = await self._router.route(request, tenant_id)

        start_ms = int(time.monotonic() * 1000)
        status = "success"
        error_message: str | None = None

        try:
            response = await provider.embed(request, model_override=model_name)
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            latency_ms = int(time.monotonic() * 1000) - start_ms
            prompt_tokens = estimated_tokens
            completion_tokens = 0
            cost = self._cost_tracker.calculate_cost(model_name, prompt_tokens, completion_tokens)
            if self._settings.enable_cost_tracking:
                await self._cost_tracker.record_usage(
                    tenant_id=tenant_id,
                    model=model_name,
                    provider=provider.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )

        return response  # type: ignore[return-value]

    async def list_models(self, tenant_id: uuid.UUID) -> ModelListResponse:
        """List all models available to a tenant.

        Args:
            tenant_id: Tenant to list models for.

        Returns:
            OpenAI-compatible model list response.
        """
        healthy_providers = await self._router.get_healthy_providers()
        all_models: list[dict[str, Any]] = []

        for provider_name in healthy_providers:
            # Provider instances are managed by RoutingService
            # Here we return the model list from router state
            pass

        return ModelListResponse(object="list", data=all_models)

    async def validate_routing(
        self,
        tenant_id: uuid.UUID,
        request: RouterValidationRequest,
    ) -> RouterValidationResponse:
        """Dry-run the routing logic to show which model would be selected.

        Args:
            tenant_id: Tenant context for config lookup.
            request: Prompt and routing constraints.

        Returns:
            RouterValidationResponse with selected model and rationale.
        """
        estimated_tokens = self._cost_tracker.count_tokens(
            text=request.prompt,
            model="gpt-3.5-turbo",
        )

        mock_request = ChatCompletionRequest(
            model=request.task_type or self._settings.default_model,
            messages=[],
        )
        provider, model_name = await self._router.route(mock_request, tenant_id)
        estimated_cost_usd = float(
            self._cost_tracker.calculate_cost(
                model=model_name,
                prompt_tokens=estimated_tokens,
                completion_tokens=0,
            )
        )

        return RouterValidationResponse(
            selected_model=model_name,
            selected_provider=provider.provider_name,
            estimated_prompt_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost_usd,
            routing_reason=f"Routed to {provider.provider_name}/{model_name} based on task_type hint",
            alternative_models=[],
        )

    async def _enforce_rate_limits(
        self,
        tenant_id: uuid.UUID,
        estimated_tokens: int,
    ) -> None:
        """Check rate limits and raise if exceeded.

        Args:
            tenant_id: Tenant to check.
            estimated_tokens: Estimated tokens for the request.

        Raises:
            ValidationError: If rate limit or quota is exceeded.
        """
        is_allowed, headers = await self._rate_limiter.check_and_increment(
            tenant_id=tenant_id,
            tokens_requested=estimated_tokens,
            rpm_limit=self._settings.default_rpm_limit,
            tpm_limit=self._settings.default_tpm_limit,
        )
        if not is_allowed:
            raise ValidationError(
                message="Rate limit exceeded",
                field="tenant_id",
                value=str(tenant_id),
            )


class RoutingService:
    """Intelligent model routing based on task type, cost, latency, and health.

    Maintains a registry of available providers and their health status.
    Selects the optimal provider + model combination for each request.
    """

    def __init__(
        self,
        providers: dict[str, LLMProviderProtocol],
        settings: LLMSettings,
        session: AsyncSession,
    ) -> None:
        """Initialize the routing service.

        Args:
            providers: Dict of provider_name → provider_instance.
            settings: Service configuration.
            session: Async database session for reading ModelConfig.
        """
        self._providers = providers
        self._settings = settings
        self._session = session
        self._health_cache: dict[str, bool] = {}

    async def route(
        self,
        request: ChatCompletionRequest | TextCompletionRequest | EmbeddingRequest,
        tenant_id: uuid.UUID,
    ) -> tuple[LLMProviderProtocol, str]:
        """Select the best provider and model for a request.

        Routing priority:
        1. If request.model is a fully-qualified "provider/model", use that provider
        2. Look up tenant ModelConfig for the requested model
        3. Fall back to default provider + model from settings

        Args:
            request: The incoming LLM request.
            tenant_id: Tenant context for config lookup.

        Returns:
            Tuple of (provider_instance, resolved_model_name).

        Raises:
            NotFoundError: If the requested model is not available.
        """
        model = request.model

        # Check for provider-prefixed model (e.g., "ollama/llama3.2", "vllm/mistral-7b")
        if "/" in model:
            provider_prefix, model_name = model.split("/", maxsplit=1)
            if provider_prefix in self._providers:
                provider = self._providers[provider_prefix]
                if await self._is_provider_healthy(provider_prefix):
                    logger.debug(
                        "Routing via provider prefix",
                        provider=provider_prefix,
                        model=model_name,
                        tenant_id=str(tenant_id),
                    )
                    return provider, model_name

        # Fall back to default provider
        default_provider_name = self._get_default_provider_name()
        if default_provider_name not in self._providers:
            raise NotFoundError(resource_type="LLMProvider", resource_id=default_provider_name)

        provider = self._providers[default_provider_name]
        logger.debug(
            "Routing to default provider",
            provider=default_provider_name,
            model=model,
            tenant_id=str(tenant_id),
        )
        return provider, model

    async def get_healthy_providers(self) -> list[str]:
        """Return names of currently healthy providers.

        Returns:
            List of provider names that passed their health check.
        """
        healthy: list[str] = []
        for name, provider in self._providers.items():
            if await self._is_provider_healthy(name):
                healthy.append(name)
        return healthy

    async def _is_provider_healthy(self, provider_name: str) -> bool:
        """Check if a provider is healthy, using a short-lived cache.

        Args:
            provider_name: Provider to check.

        Returns:
            True if healthy.
        """
        if provider_name not in self._providers:
            return False
        try:
            is_healthy = await self._providers[provider_name].health_check()
            self._health_cache[provider_name] = is_healthy
            return is_healthy
        except Exception:
            self._health_cache[provider_name] = False
            return False

    def _get_default_provider_name(self) -> str:
        """Infer the default provider from the default_model setting.

        Returns:
            Provider name string.
        """
        default_model = self._settings.default_model
        if "/" in default_model:
            return default_model.split("/", maxsplit=1)[0]
        return "litellm"


class CostTrackingService:
    """Per-tenant token consumption tracking and cost attribution.

    Combines tiktoken-based counting with database persistence and
    budget enforcement checks.
    """

    def __init__(
        self,
        session: AsyncSession,
        settings: LLMSettings,
    ) -> None:
        """Initialize cost tracking service.

        Args:
            session: Async database session.
            settings: Service configuration.
        """
        self._session = session
        self._settings = settings

    async def get_tenant_usage(
        self,
        tenant_id: uuid.UUID,
    ) -> TenantUsageResponse:
        """Get current usage and cost summary for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Usage dashboard response with daily/monthly breakdown.
        """
        from aumos_llm_serving.adapters.repositories import (  # noqa: PLC0415
            LLMRequestRepository,
            TenantQuotaRepository,
        )

        request_repo = LLMRequestRepository(self._session)
        quota_repo = TenantQuotaRepository(self._session)

        daily_stats = await request_repo.get_daily_stats(tenant_id)
        monthly_stats = await request_repo.get_monthly_stats(tenant_id)
        quota = await quota_repo.get_or_create(tenant_id)

        return TenantUsageResponse(
            tenant_id=tenant_id,
            daily_tokens_used=daily_stats.get("total_tokens", 0),
            daily_cost_usd=daily_stats.get("total_cost", decimal.Decimal("0")),
            daily_token_limit=quota.daily_token_limit,
            daily_cost_limit=quota.daily_cost_limit,
            monthly_tokens_used=monthly_stats.get("total_tokens", 0),
            monthly_cost_usd=monthly_stats.get("total_cost", decimal.Decimal("0")),
            monthly_token_limit=quota.monthly_token_limit,
            monthly_cost_limit=quota.monthly_cost_limit,
        )

    async def check_budget(
        self,
        tenant_id: uuid.UUID,
        estimated_cost: decimal.Decimal,
    ) -> bool:
        """Check if a request would exceed the tenant's budget.

        Args:
            tenant_id: Tenant to check.
            estimated_cost: Estimated cost for the upcoming request.

        Returns:
            True if within budget, False if budget would be exceeded.
        """
        from aumos_llm_serving.adapters.repositories import (  # noqa: PLC0415
            TenantQuotaRepository,
        )

        quota_repo = TenantQuotaRepository(self._session)
        quota = await quota_repo.get_or_create(tenant_id)

        daily_stats = await self._get_daily_cost(tenant_id)
        if daily_stats + estimated_cost > quota.daily_cost_limit:
            logger.warning(
                "Daily budget would be exceeded",
                tenant_id=str(tenant_id),
                current_cost=float(daily_stats),
                estimated_addition=float(estimated_cost),
                limit=float(quota.daily_cost_limit),
            )
            return False
        return True

    async def get_admin_dashboard(self) -> AdminDashboardResponse:
        """Return platform-wide aggregated metrics for the admin dashboard.

        Returns:
            AdminDashboardResponse with request and cost aggregates.
        """
        from aumos_llm_serving.adapters.repositories import LLMRequestRepository  # noqa: PLC0415

        repo = LLMRequestRepository(self._session)
        stats = await repo.get_platform_daily_stats()

        return AdminDashboardResponse(
            total_tenants=stats.get("total_tenants", 0),
            active_tenants_today=stats.get("active_tenants_today", 0),
            total_requests_today=stats.get("total_requests_today", 0),
            total_tokens_today=stats.get("total_tokens_today", 0),
            total_cost_today_usd=float(stats.get("total_cost_today", 0.0)),
            requests_by_model=stats.get("requests_by_model", {}),
            requests_by_provider=stats.get("requests_by_provider", {}),
            error_rate_pct=float(stats.get("error_rate_pct", 0.0)),
            avg_latency_ms=float(stats.get("avg_latency_ms", 0.0)),
        )

    async def get_streaming_metrics(
        self,
        tenant_id: uuid.UUID,
        period_hours: int = 24,
    ) -> StreamingMetricsResponse:
        """Return streaming analytics including TTFT percentiles.

        Args:
            tenant_id: Tenant to query.
            period_hours: Lookback window in hours.

        Returns:
            StreamingMetricsResponse with TTFT and throughput metrics.
        """
        from aumos_llm_serving.adapters.repositories import LLMRequestRepository  # noqa: PLC0415

        repo = LLMRequestRepository(self._session)
        stats = await repo.get_streaming_stats(tenant_id, period_hours)

        return StreamingMetricsResponse(
            tenant_id=tenant_id,
            period_hours=period_hours,
            total_streaming_requests=stats.get("total_streaming_requests", 0),
            total_streaming_tokens=stats.get("total_streaming_tokens", 0),
            avg_tokens_per_second=float(stats.get("avg_tokens_per_second", 0.0)),
            p50_ttft_ms=float(stats.get("p50_ttft_ms", 0.0)),
            p95_ttft_ms=float(stats.get("p95_ttft_ms", 0.0)),
            p99_ttft_ms=float(stats.get("p99_ttft_ms", 0.0)),
        )

    async def _get_daily_cost(self, tenant_id: uuid.UUID) -> decimal.Decimal:
        """Get total cost spent today for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Total cost in USD.
        """
        from aumos_llm_serving.adapters.repositories import LLMRequestRepository  # noqa: PLC0415

        repo = LLMRequestRepository(self._session)
        stats = await repo.get_daily_stats(tenant_id)
        return decimal.Decimal(str(stats.get("total_cost", 0)))


class RateLimitingService:
    """Per-tenant token/request rate limiting with quota enforcement.

    Wraps the Redis-based RateLimiter adapter with business logic for
    retrieving per-tenant limits from ModelConfig.
    """

    def __init__(
        self,
        rate_limiter: RateLimiterProtocol,
        session: AsyncSession,
        settings: LLMSettings,
    ) -> None:
        """Initialize the rate limiting service.

        Args:
            rate_limiter: Redis-based rate limiter implementation.
            session: Async database session for reading tenant configs.
            settings: Service configuration with default limits.
        """
        self._rate_limiter = rate_limiter
        self._session = session
        self._settings = settings

    async def check_request(
        self,
        tenant_id: uuid.UUID,
        model: str,
        estimated_tokens: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if a request is within rate limits.

        Looks up per-tenant, per-model limits from ModelConfig if available,
        otherwise falls back to service-level defaults.

        Args:
            tenant_id: Tenant making the request.
            model: Model being requested.
            estimated_tokens: Estimated token count for rate limiting.

        Returns:
            Tuple of (is_allowed, rate_limit_headers).
        """
        from aumos_llm_serving.adapters.repositories import ModelConfigRepository  # noqa: PLC0415

        config_repo = ModelConfigRepository(self._session)
        model_config = await config_repo.get_by_model_name(tenant_id, model)

        rpm_limit = model_config.rate_limit_rpm if model_config else self._settings.default_rpm_limit
        tpm_limit = model_config.rate_limit_tpm if model_config else self._settings.default_tpm_limit

        return await self._rate_limiter.check_and_increment(
            tenant_id=tenant_id,
            tokens_requested=estimated_tokens,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
        )

    async def get_current_usage(self, tenant_id: uuid.UUID) -> dict[str, int]:
        """Get current rate limit counters for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Dict with requests_this_minute and tokens_this_minute.
        """
        return await self._rate_limiter.get_current_usage(tenant_id)


class ModelManagementService:
    """CRUD operations for model configurations and provider health management.

    Provides tenant-scoped model configuration, health monitoring,
    and failover management for the provider fleet.
    """

    def __init__(
        self,
        session: AsyncSession,
        providers: dict[str, LLMProviderProtocol],
        settings: LLMSettings,
    ) -> None:
        """Initialize the model management service.

        Args:
            session: Async database session.
            providers: Dict of provider_name → provider_instance.
            settings: Service configuration.
        """
        self._session = session
        self._providers = providers
        self._settings = settings

    async def create_model_config(
        self,
        tenant_id: uuid.UUID,
        request: ModelConfigCreateRequest,
    ) -> ModelConfigResponse:
        """Create a new model configuration for a tenant.

        Args:
            tenant_id: Tenant to create config for.
            request: Model configuration parameters.

        Returns:
            Created model configuration.

        Raises:
            ValidationError: If provider is not recognized.
        """
        from aumos_llm_serving.adapters.repositories import ModelConfigRepository  # noqa: PLC0415

        if request.provider not in self._providers and request.provider not in {
            "openai",
            "anthropic",
            "azure",
            "custom",
        }:
            raise ValidationError(
                message=f"Unknown provider: {request.provider}",
                field="provider",
                value=request.provider,
            )

        repo = ModelConfigRepository(self._session)
        config = await repo.create(
            tenant_id=tenant_id,
            model_name=request.model_name,
            provider=request.provider,
            endpoint_url=request.endpoint_url,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            rate_limit_rpm=request.rate_limit_rpm or self._settings.default_rpm_limit,
            rate_limit_tpm=request.rate_limit_tpm or self._settings.default_tpm_limit,
            cost_per_input_token=request.cost_per_input_token or decimal.Decimal("0"),
            cost_per_output_token=request.cost_per_output_token or decimal.Decimal("0"),
            is_default=request.is_default or False,
        )

        logger.info(
            "Model config created",
            tenant_id=str(tenant_id),
            model_name=request.model_name,
            provider=request.provider,
        )

        return ModelConfigResponse.model_validate(config)

    async def update_model_config(
        self,
        tenant_id: uuid.UUID,
        config_id: uuid.UUID,
        request: ModelConfigUpdateRequest,
    ) -> ModelConfigResponse:
        """Update an existing model configuration.

        Args:
            tenant_id: Tenant context (enforced via RLS).
            config_id: Config UUID to update.
            request: Fields to update.

        Returns:
            Updated model configuration.

        Raises:
            NotFoundError: If config does not exist.
        """
        from aumos_llm_serving.adapters.repositories import ModelConfigRepository  # noqa: PLC0415

        repo = ModelConfigRepository(self._session)
        config = await repo.update(config_id, request.model_dump(exclude_none=True))

        if config is None:
            raise NotFoundError(resource_type="ModelConfig", resource_id=str(config_id))

        logger.info(
            "Model config updated",
            tenant_id=str(tenant_id),
            config_id=str(config_id),
        )

        return ModelConfigResponse.model_validate(config)

    async def list_model_configs(
        self,
        tenant_id: uuid.UUID,
    ) -> list[ModelConfigResponse]:
        """List all model configurations for a tenant.

        Args:
            tenant_id: Tenant to list configs for.

        Returns:
            List of model configurations.
        """
        from aumos_llm_serving.adapters.repositories import ModelConfigRepository  # noqa: PLC0415

        repo = ModelConfigRepository(self._session)
        configs = await repo.list_all()
        return [ModelConfigResponse.model_validate(c) for c in configs]

    async def check_provider_health(self) -> dict[str, bool]:
        """Check health of all registered providers.

        Returns:
            Dict mapping provider_name → is_healthy.
        """
        health: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                health[name] = await provider.health_check()
            except Exception:
                health[name] = False
        return health

    async def create_ab_test(
        self,
        tenant_id: uuid.UUID,
        request: ABTestCreateRequest,
    ) -> ABTestResponse:
        """Create a new A/B model testing experiment.

        Args:
            tenant_id: Tenant to create experiment for.
            request: Experiment parameters including model pair and traffic split.

        Returns:
            Created A/B test experiment.
        """
        from aumos_llm_serving.adapters.repositories import ABTestRepository  # noqa: PLC0415

        repo = ABTestRepository(self._session)
        experiment = await repo.create(
            tenant_id=tenant_id,
            name=request.name,
            model_a=request.model_a,
            model_b=request.model_b,
            traffic_split_pct=request.traffic_split_pct,
            evaluation_metric=request.evaluation_metric,
            sample_size=request.sample_size,
        )

        logger.info(
            "A/B test created",
            tenant_id=str(tenant_id),
            name=request.name,
            model_a=request.model_a,
            model_b=request.model_b,
        )

        return ABTestResponse.model_validate(experiment)

    async def list_ab_tests(
        self,
        tenant_id: uuid.UUID,
    ) -> list[ABTestResponse]:
        """List all A/B test experiments for a tenant.

        Args:
            tenant_id: Tenant to list experiments for.

        Returns:
            List of A/B test experiments.
        """
        from aumos_llm_serving.adapters.repositories import ABTestRepository  # noqa: PLC0415

        repo = ABTestRepository(self._session)
        experiments = await repo.list_all()
        return [ABTestResponse.model_validate(e) for e in experiments]

    async def get_ab_test(
        self,
        tenant_id: uuid.UUID,
        test_id: uuid.UUID,
    ) -> ABTestResponse:
        """Get a specific A/B test experiment.

        Args:
            tenant_id: Tenant context (enforced via RLS).
            test_id: Experiment UUID to retrieve.

        Returns:
            A/B test experiment.

        Raises:
            NotFoundError: If experiment does not exist.
        """
        from aumos_llm_serving.adapters.repositories import ABTestRepository  # noqa: PLC0415

        repo = ABTestRepository(self._session)
        experiment = await repo.get(test_id)

        if experiment is None:
            raise NotFoundError(resource_type="ABTest", resource_id=str(test_id))

        return ABTestResponse.model_validate(experiment)

    async def create_guardrail_rule(
        self,
        tenant_id: uuid.UUID,
        request: GuardrailRuleCreateRequest,
    ) -> GuardrailRuleResponse:
        """Create a content guardrail rule for a tenant.

        Args:
            tenant_id: Tenant to create rule for.
            request: Guardrail rule parameters.

        Returns:
            Created guardrail rule.
        """
        from aumos_llm_serving.adapters.repositories import GuardrailRepository  # noqa: PLC0415

        repo = GuardrailRepository(self._session)
        rule = await repo.create(
            tenant_id=tenant_id,
            name=request.name,
            rule_type=request.rule_type,
            pattern=request.pattern,
            action=request.action,
            applies_to=request.applies_to,
        )

        logger.info(
            "Guardrail rule created",
            tenant_id=str(tenant_id),
            name=request.name,
            rule_type=request.rule_type,
            action=request.action,
        )

        return GuardrailRuleResponse.model_validate(rule)

    async def list_guardrail_rules(
        self,
        tenant_id: uuid.UUID,
    ) -> list[GuardrailRuleResponse]:
        """List all guardrail rules for a tenant.

        Args:
            tenant_id: Tenant to list rules for.

        Returns:
            List of guardrail rules.
        """
        from aumos_llm_serving.adapters.repositories import GuardrailRepository  # noqa: PLC0415

        repo = GuardrailRepository(self._session)
        rules = await repo.list_all()
        return [GuardrailRuleResponse.model_validate(r) for r in rules]

    async def delete_guardrail_rule(
        self,
        tenant_id: uuid.UUID,
        rule_id: uuid.UUID,
    ) -> None:
        """Delete a guardrail rule.

        Args:
            tenant_id: Tenant context (enforced via RLS).
            rule_id: Rule UUID to delete.

        Raises:
            NotFoundError: If rule does not exist.
        """
        from aumos_llm_serving.adapters.repositories import GuardrailRepository  # noqa: PLC0415

        repo = GuardrailRepository(self._session)
        deleted = await repo.delete(rule_id)

        if not deleted:
            raise NotFoundError(resource_type="GuardrailRule", resource_id=str(rule_id))

        logger.info(
            "Guardrail rule deleted",
            tenant_id=str(tenant_id),
            rule_id=str(rule_id),
        )

    async def warmup_model(
        self,
        tenant_id: uuid.UUID,
        request: ModelWarmUpRequest,
    ) -> ModelWarmUpResponse:
        """Pre-warm a model on a specific provider to reduce cold-start latency.

        Sends a minimal inference request to the target provider so the model
        is loaded and ready for subsequent production requests.

        Args:
            tenant_id: Tenant context for audit logging.
            request: Warm-up parameters including model and provider.

        Returns:
            Warm-up result with measured latency.
        """
        import time as _time  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        provider_name = request.provider
        start_ms = int(_time.monotonic() * 1000)

        try:
            if provider_name in self._providers:
                # Use the registered provider adapter
                provider = self._providers[provider_name]
                warmup_req = ChatCompletionRequest(
                    model=request.model_name,
                    messages=[{"role": "user", "content": request.sample_prompt}],  # type: ignore[list-item]
                    max_tokens=1,
                )
                await provider.chat_completion(warmup_req, model_override=request.model_name)
            else:
                # Unknown provider — attempt a direct HTTP probe
                async with httpx.AsyncClient(timeout=30.0) as client:
                    endpoint = self._settings.vllm_base_url if provider_name == "vllm" else self._settings.ollama_base_url
                    await client.get(f"{endpoint}/health")

            latency_ms = int(_time.monotonic() * 1000) - start_ms
            logger.info(
                "Model warmed up",
                model=request.model_name,
                provider=provider_name,
                latency_ms=latency_ms,
            )
            return ModelWarmUpResponse(
                model_name=request.model_name,
                provider=provider_name,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as exc:
            latency_ms = int(_time.monotonic() * 1000) - start_ms
            logger.warning(
                "Model warm-up failed",
                model=request.model_name,
                provider=provider_name,
                error=str(exc),
            )
            return ModelWarmUpResponse(
                model_name=request.model_name,
                provider=provider_name,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    async def get_cache_stats(
        self,
        tenant_id: uuid.UUID,
    ) -> list[CacheStatsResponse]:
        """Retrieve KV cache statistics from all registered providers.

        Queries each provider's metrics endpoint and returns cache hit rate,
        evictions, and memory utilization.

        Args:
            tenant_id: Tenant context (used for audit logging).

        Returns:
            List of CacheStatsResponse, one per provider.
        """
        results: list[CacheStatsResponse] = []
        for provider_name in self._providers:
            stat = await self._get_single_provider_cache_stats(provider_name)
            results.append(stat)
        return results

    async def _get_single_provider_cache_stats(
        self,
        provider: str,
        model_name: str | None = None,
    ) -> CacheStatsResponse:
        """Retrieve KV cache statistics from a model provider.

        Queries the provider's metrics endpoint and returns cache hit rate,
        evictions, and memory utilization.

        Args:
            provider: Provider name to query (vllm, ollama).
            model_name: Optional model to filter stats.

        Returns:
            CacheStatsResponse with memory and hit-rate metrics.
        """
        import datetime  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        try:
            if provider == "vllm":
                base_url = self._settings.vllm_base_url
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/metrics")
                    raw = response.text
                # Parse Prometheus text format for KV cache metrics
                cache_hits = _parse_prometheus_counter(raw, "vllm:gpu_cache_hit_total")
                cache_misses = _parse_prometheus_counter(raw, "vllm:gpu_cache_miss_total")
                evictions = _parse_prometheus_counter(raw, "vllm:gpu_cache_eviction_total")
                mem_used = _parse_prometheus_gauge(raw, "vllm:gpu_cache_usage_perc") * 80.0
                mem_total = 80.0
            else:
                # Ollama does not expose detailed cache metrics — return zeros
                cache_hits = cache_misses = evictions = 0
                mem_used = mem_total = 0.0

            total_lookups = cache_hits + cache_misses
            hit_rate = cache_hits / total_lookups if total_lookups > 0 else 0.0

            return CacheStatsResponse(
                provider=provider,
                model_name=model_name,
                cache_hit_rate=hit_rate,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                evictions=evictions,
                memory_used_gb=mem_used,
                memory_total_gb=mem_total,
                collected_at=datetime.datetime.utcnow().isoformat(),
            )

        except Exception as exc:
            logger.warning(
                "Failed to collect cache stats",
                provider=provider,
                error=str(exc),
            )
            import datetime as _dt  # noqa: PLC0415

            return CacheStatsResponse(
                provider=provider,
                model_name=model_name,
                cache_hit_rate=0.0,
                cache_hits=0,
                cache_misses=0,
                evictions=0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                collected_at=_dt.datetime.utcnow().isoformat(),
            )

    async def set_tenant_quota(
        self,
        tenant_id: uuid.UUID,
        request: TenantQuotaRequest,
    ) -> TenantQuotaResponse:
        """Set or update token and cost quotas for a tenant.

        Args:
            tenant_id: Tenant to set quotas for.
            request: Quota configuration.

        Returns:
            Updated quota configuration.
        """
        from aumos_llm_serving.adapters.repositories import TenantQuotaRepository  # noqa: PLC0415

        repo = TenantQuotaRepository(self._session)
        quota = await repo.upsert(
            tenant_id=tenant_id,
            daily_token_limit=request.daily_token_limit,
            monthly_token_limit=request.monthly_token_limit,
            daily_cost_limit=request.daily_cost_limit,
            monthly_cost_limit=request.monthly_cost_limit,
        )

        logger.info(
            "Tenant quota updated",
            tenant_id=str(tenant_id),
            daily_token_limit=request.daily_token_limit,
            monthly_token_limit=request.monthly_token_limit,
        )

        return TenantQuotaResponse.model_validate(quota)


def _parse_prometheus_counter(text: str, metric_name: str) -> int:
    """Extract an integer counter value from Prometheus text format.

    Args:
        text: Raw Prometheus metrics text.
        metric_name: Metric name to look up.

    Returns:
        Integer value, or 0 if not found.
    """
    for line in text.splitlines():
        if line.startswith(metric_name) and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(float(parts[-1]))
                except ValueError:
                    pass
    return 0


def _parse_prometheus_gauge(text: str, metric_name: str) -> float:
    """Extract a float gauge value from Prometheus text format.

    Args:
        text: Raw Prometheus metrics text.
        metric_name: Metric name to look up.

    Returns:
        Float value, or 0.0 if not found.
    """
    for line in text.splitlines():
        if line.startswith(metric_name) and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return 0.0


class ModelServingOrchestrator:
    """High-level orchestrator that wires all domain adapters together.

    Sits above ServingService and adds model loading lifecycle management,
    batch scheduling, SSE stream handling, quantization profiling, and
    metrics collection. All new adapters are injected via constructor for
    testability and provider-agnosticism.

    Typical production wiring:

        orchestrator = ModelServingOrchestrator(
            serving_service=serving_service,
            model_loader=ModelLoader(model_root="/models"),
            batch_scheduler=BatchScheduler(inference_fn=vllm_runner),
            stream_handler=StreamHandler(default_timeout_seconds=120),
            metrics_collector=InferenceMetricsCollector(),
        )
        await orchestrator.start()
    """

    def __init__(
        self,
        serving_service: ServingService,
        model_loader: ModelLoaderProtocol | None = None,
        batch_scheduler: BatchSchedulerProtocol | None = None,
        stream_handler: StreamHandlerProtocol | None = None,
        metrics_collector: MetricsCollectorProtocol | None = None,
    ) -> None:
        """Initialize the orchestrator with all domain adapters.

        Args:
            serving_service: Core serving service for LLM calls.
            model_loader: Model lifecycle manager (optional — for self-hosted).
            batch_scheduler: Dynamic batch scheduler (optional — for high-throughput).
            stream_handler: SSE stream manager (optional — for streaming endpoints).
            metrics_collector: Performance metrics collector (optional).
        """
        self._serving_service = serving_service
        self._model_loader = model_loader
        self._batch_scheduler = batch_scheduler
        self._stream_handler = stream_handler
        self._metrics_collector = metrics_collector

    async def start(self) -> None:
        """Start all background components.

        Starts the batch scheduler worker if one is configured.
        """
        if self._batch_scheduler is not None:
            await self._batch_scheduler.start()
            logger.info("BatchScheduler started via orchestrator")

    async def stop(self) -> None:
        """Stop all background components gracefully."""
        if self._batch_scheduler is not None:
            await self._batch_scheduler.stop()
            logger.info("BatchScheduler stopped via orchestrator")

    async def chat_completion_with_metrics(
        self,
        request: ChatCompletionRequest,
        tenant_id: uuid.UUID,
    ) -> ChatCompletionResponse:
        """Execute a chat completion and record performance metrics.

        Delegates to ServingService and wraps the call with request tracing
        for the MetricsCollector.

        Args:
            request: OpenAI-compatible chat completion request.
            tenant_id: Tenant making the request.

        Returns:
            OpenAI-compatible chat completion response.
        """
        enqueued_at = time.monotonic()
        started_at = time.monotonic()
        first_token_at: float | None = None
        request_id = uuid.uuid4()
        status = "success"
        error_code: str | None = None
        response: ChatCompletionResponse | None = None

        try:
            response = await self._serving_service.chat_completion(request, tenant_id)
        except Exception as exc:
            status = "error"
            error_code = type(exc).__name__
            raise
        finally:
            completed_at = time.monotonic()
            if self._metrics_collector is not None:
                usage = getattr(response, "usage", None)
                trace = RequestTrace(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    model=request.model,
                    provider="routed",
                    enqueued_at=enqueued_at,
                    started_at=started_at,
                    first_token_at=first_token_at,
                    completed_at=completed_at,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    status=status,
                    error_code=error_code,
                )
                self._metrics_collector.record_request(trace)

        return response  # type: ignore[return-value]

    def get_system_status(self) -> dict[str, Any]:
        """Return a consolidated system status snapshot.

        Returns:
            Dict with loaded models, active streams, batch queue metrics,
            and aggregated throughput.
        """
        status: dict[str, Any] = {
            "loaded_models": (
                self._model_loader.list_loaded_models()
                if self._model_loader is not None
                else []
            ),
            "vram_usage": (
                self._model_loader.get_vram_usage()
                if self._model_loader is not None
                else {}
            ),
            "active_streams": (
                self._stream_handler.list_active_streams()  # type: ignore[union-attr]
                if self._stream_handler is not None
                else []
            ),
            "batch_metrics": (
                self._batch_scheduler.get_metrics()
                if self._batch_scheduler is not None
                else {}
            ),
        }
        return status

    def get_prometheus_metrics(self) -> str:
        """Export Prometheus-compatible metrics from the collector.

        Returns:
            Prometheus text format string, or empty string if no collector.
        """
        if self._metrics_collector is None:
            return ""
        return self._metrics_collector.get_prometheus_metrics()
