"""FastAPI router for aumos-llm-serving.

Two namespaces:
- /v1/* — OpenAI-compatible endpoints (drop-in replacement)
- /serving/* — AumOS extension endpoints (model management, quotas, usage)
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session

from aumos_llm_serving.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelConfigCreateRequest,
    ModelConfigResponse,
    ModelConfigUpdateRequest,
    ModelListResponse,
    TenantQuotaRequest,
    TenantQuotaResponse,
    TenantUsageResponse,
    TextCompletionRequest,
    TextCompletionResponse,
)
from aumos_llm_serving.core.services import (
    CostTrackingService,
    ModelManagementService,
    RateLimitingService,
    RoutingService,
    ServingService,
)
from aumos_llm_serving.settings import LLMSettings

router = APIRouter()
settings = LLMSettings()


# =============================================================================
# Dependency factories
# =============================================================================


def get_settings() -> LLMSettings:
    """Return the service settings singleton."""
    return settings


def get_provider_registry(request: Request) -> dict:
    """Get the provider registry from app state.

    Providers are initialized at startup and stored in app.state.
    """
    return getattr(request.app.state, "providers", {})


def get_rate_limiter(request: Request):  # type: ignore[return]
    """Get the rate limiter from app state."""
    return request.app.state.rate_limiter


async def get_serving_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    request: Request,
    svc_settings: Annotated[LLMSettings, Depends(get_settings)],
) -> ServingService:
    """Build and return a ServingService for the current request."""
    from aumos_llm_serving.adapters.cost_tracker import CostTracker  # noqa: PLC0415
    from aumos_llm_serving.adapters.model_router import ModelRouter  # noqa: PLC0415

    providers = get_provider_registry(request)
    rate_limiter = get_rate_limiter(request)

    routing_service = RoutingService(providers=providers, settings=svc_settings, session=session)
    cost_tracker = CostTracker(session=session)
    model_router = ModelRouter(routing_service=routing_service, session=session)

    return ServingService(
        router=model_router,
        cost_tracker=cost_tracker,
        rate_limiter=rate_limiter,
        session=session,
        settings=svc_settings,
    )


async def get_model_management_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    request: Request,
    svc_settings: Annotated[LLMSettings, Depends(get_settings)],
) -> ModelManagementService:
    """Build and return a ModelManagementService for the current request."""
    providers = get_provider_registry(request)
    return ModelManagementService(session=session, providers=providers, settings=svc_settings)


async def get_cost_tracking_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    svc_settings: Annotated[LLMSettings, Depends(get_settings)],
) -> CostTrackingService:
    """Build and return a CostTrackingService for the current request."""
    return CostTrackingService(session=session, settings=svc_settings)


# =============================================================================
# OpenAI-Compatible: /v1/* endpoints
# =============================================================================


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["OpenAI Compatible"],
    summary="Create chat completion",
    description="OpenAI-compatible chat completion. Supports all major models via provider routing.",
)
async def create_chat_completion(
    request_body: ChatCompletionRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ServingService, Depends(get_serving_service)],
) -> ChatCompletionResponse:
    """Create a chat completion response.

    Args:
        request_body: OpenAI-compatible chat completion request.
        tenant: Tenant context from auth middleware.
        service: Serving service instance.

    Returns:
        OpenAI-compatible chat completion response.
    """
    return await service.chat_completion(
        request=request_body,
        tenant_id=tenant.tenant_id,
    )


@router.post(
    "/v1/completions",
    response_model=TextCompletionResponse,
    tags=["OpenAI Compatible"],
    summary="Create text completion",
    description="OpenAI-compatible text completion (legacy endpoint).",
)
async def create_text_completion(
    request_body: TextCompletionRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ServingService, Depends(get_serving_service)],
) -> TextCompletionResponse:
    """Create a text completion response.

    Args:
        request_body: OpenAI-compatible text completion request.
        tenant: Tenant context from auth middleware.
        service: Serving service instance.

    Returns:
        OpenAI-compatible text completion response.
    """
    return await service.text_completion(
        request=request_body,
        tenant_id=tenant.tenant_id,
    )


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    tags=["OpenAI Compatible"],
    summary="Create embeddings",
    description="OpenAI-compatible embedding generation.",
)
async def create_embeddings(
    request_body: EmbeddingRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ServingService, Depends(get_serving_service)],
) -> EmbeddingResponse:
    """Generate embeddings for input text.

    Args:
        request_body: OpenAI-compatible embedding request.
        tenant: Tenant context from auth middleware.
        service: Serving service instance.

    Returns:
        OpenAI-compatible embedding response.
    """
    return await service.embed(
        request=request_body,
        tenant_id=tenant.tenant_id,
    )


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    tags=["OpenAI Compatible"],
    summary="List available models",
    description="OpenAI-compatible model listing. Returns models available to this tenant.",
)
async def list_models(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ServingService, Depends(get_serving_service)],
) -> ModelListResponse:
    """List models available to the requesting tenant.

    Args:
        tenant: Tenant context from auth middleware.
        service: Serving service instance.

    Returns:
        OpenAI-compatible model list response.
    """
    return await service.list_models(tenant_id=tenant.tenant_id)


# =============================================================================
# AumOS Extensions: /serving/* endpoints
# =============================================================================


@router.get(
    "/serving/usage/{tenant_id}",
    response_model=TenantUsageResponse,
    tags=["AumOS LLM Serving"],
    summary="Get tenant usage dashboard",
    description="Returns daily and monthly token consumption and cost for a tenant.",
)
async def get_tenant_usage(
    tenant_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[CostTrackingService, Depends(get_cost_tracking_service)],
) -> TenantUsageResponse:
    """Get usage and cost dashboard for a tenant.

    Args:
        tenant_id: Tenant UUID to get usage for.
        tenant: Authenticated tenant context.
        service: Cost tracking service instance.

    Returns:
        Usage dashboard with daily/monthly breakdown.
    """
    return await service.get_tenant_usage(tenant_id=tenant_id)


@router.post(
    "/serving/models",
    response_model=ModelConfigResponse,
    status_code=201,
    tags=["AumOS LLM Serving"],
    summary="Configure a model",
    description="Add a model configuration for this tenant with custom routing and cost parameters.",
)
async def create_model_config(
    request_body: ModelConfigCreateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ModelManagementService, Depends(get_model_management_service)],
) -> ModelConfigResponse:
    """Create a new model configuration for the tenant.

    Args:
        request_body: Model configuration parameters.
        tenant: Authenticated tenant context.
        service: Model management service instance.

    Returns:
        Created model configuration.
    """
    return await service.create_model_config(
        tenant_id=tenant.tenant_id,
        request=request_body,
    )


@router.put(
    "/serving/models/{config_id}",
    response_model=ModelConfigResponse,
    tags=["AumOS LLM Serving"],
    summary="Update model configuration",
    description="Update an existing model configuration.",
)
async def update_model_config(
    config_id: uuid.UUID,
    request_body: ModelConfigUpdateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ModelManagementService, Depends(get_model_management_service)],
) -> ModelConfigResponse:
    """Update an existing model configuration.

    Args:
        config_id: UUID of the model config to update.
        request_body: Fields to update.
        tenant: Authenticated tenant context.
        service: Model management service instance.

    Returns:
        Updated model configuration.
    """
    return await service.update_model_config(
        tenant_id=tenant.tenant_id,
        config_id=config_id,
        request=request_body,
    )


@router.get(
    "/serving/models",
    response_model=list[ModelConfigResponse],
    tags=["AumOS LLM Serving"],
    summary="List model configurations",
    description="List all model configurations for this tenant.",
)
async def list_model_configs(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ModelManagementService, Depends(get_model_management_service)],
) -> list[ModelConfigResponse]:
    """List model configurations for the tenant.

    Args:
        tenant: Authenticated tenant context.
        service: Model management service instance.

    Returns:
        List of model configurations.
    """
    return await service.list_model_configs(tenant_id=tenant.tenant_id)


@router.post(
    "/serving/quotas",
    response_model=TenantQuotaResponse,
    tags=["AumOS LLM Serving"],
    summary="Set tenant quota",
    description="Set or update token and cost quotas for a tenant.",
)
async def set_tenant_quota(
    request_body: TenantQuotaRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ModelManagementService, Depends(get_model_management_service)],
) -> TenantQuotaResponse:
    """Set or update quotas for the tenant.

    Args:
        request_body: Quota configuration.
        tenant: Authenticated tenant context.
        service: Model management service instance.

    Returns:
        Updated quota configuration.
    """
    return await service.set_tenant_quota(
        tenant_id=tenant.tenant_id,
        request=request_body,
    )


@router.get(
    "/serving/health/providers",
    response_model=dict[str, bool],
    tags=["AumOS LLM Serving"],
    summary="Provider health status",
    description="Check health of all registered LLM providers.",
)
async def get_provider_health(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ModelManagementService, Depends(get_model_management_service)],
) -> dict[str, bool]:
    """Check health of all registered providers.

    Args:
        tenant: Authenticated tenant context.
        service: Model management service instance.

    Returns:
        Dict mapping provider_name to is_healthy boolean.
    """
    return await service.check_provider_health()
