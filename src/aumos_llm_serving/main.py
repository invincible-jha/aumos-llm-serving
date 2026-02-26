"""AumOS LLM Serving service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_llm_serving.settings import LLMSettings

logger = get_logger(__name__)
settings = LLMSettings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown."""
    logger.info(
        "Starting aumos-llm-serving",
        version="0.1.0",
        environment=settings.environment,
        default_model=settings.default_model,
    )

    # Initialize database connection pool
    init_database(settings.database)
    logger.info("Database connection pool initialized")

    # Import and initialize the serving service (lazy to avoid circular imports)
    from aumos_llm_serving.adapters.rate_limiter import RateLimiter  # noqa: PLC0415

    rate_limiter = RateLimiter(settings=settings)
    await rate_limiter.connect()
    app.state.rate_limiter = rate_limiter
    logger.info("Redis rate limiter connected", url=settings.redis.url)

    yield

    # Shutdown
    await rate_limiter.disconnect()
    logger.info("aumos-llm-serving shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-llm-serving",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db),
        # HealthCheck(name="redis", check_fn=check_redis),
    ],
)

# Include OpenAI-compatible and AumOS extension routers
from aumos_llm_serving.api.router import router  # noqa: E402

app.include_router(router)
