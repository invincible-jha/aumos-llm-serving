# CLAUDE.md — AumOS LLM Serving

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-llm-serving`) is part of **Tier B: Open Core**:
Unified LLM serving infrastructure shared by all synthesis engines and the agent framework.

**Release Tier:** B: Open Core
**Product Mapping:** Shared Infrastructure — LLM Serving Layer
**Phase:** 1B (Months 6-9)

## Repo Purpose

Provides a unified, OpenAI-compatible LLM serving API that abstracts over vLLM (production
GPU serving), Ollama (development), and LiteLLM (100+ cloud providers). Every synthesis
engine and agent in AumOS routes LLM calls through this service to get consistent cost
tracking, rate limiting, and intelligent model routing.

## Architecture Position

```
aumos-platform-core → aumos-auth-gateway → aumos-llm-serving
                                         ↑ imported by:
aumos-agent-framework ────────────────────┘  (all LLM calls)
aumos-tabular-engine ──────────────────────┘  (augmentation)
aumos-text-engine ─────────────────────────┘  (generation)
aumos-image-engine ────────────────────────┘  (captioning)
aumos-data-pipeline ───────────────────────┘  (enrichment)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events

**Downstream dependents (other repos IMPORT from this):**
- `aumos-agent-framework` — all LLM inference calls
- `aumos-tabular-engine` — LLM-based data augmentation
- `aumos-text-engine` — text generation and transformation
- `aumos-image-engine` — image captioning and description
- `aumos-data-pipeline` — data enrichment tasks

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| litellm | 1.30+ | Multi-provider LLM routing (100+ providers) |
| tiktoken | 0.6+ | OpenAI-compatible token counting |
| openai | 1.12+ | OpenAI SDK (also used by LiteLLM) |
| httpx | 0.27+ | Async HTTP client for vLLM/Ollama |
| redis | 5.0+ | Rate limiting and quota state |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure

```
src/aumos_llm_serving/
├── __init__.py
├── main.py                   # FastAPI app entry point
├── settings.py               # Extends AumOSSettings
├── api/
│   ├── __init__.py
│   ├── router.py             # OpenAI-compatible + AumOS endpoints
│   └── schemas.py            # OpenAI-compatible + AumOS Pydantic models
├── core/
│   ├── __init__.py
│   ├── models.py             # LLMRequest, ModelConfig, TenantQuota
│   ├── services.py           # ServingService, RoutingService, CostTrackingService,
│   │                         #   RateLimitingService, ModelManagementService
│   └── interfaces.py         # LLMProviderProtocol, ModelRouterProtocol, etc.
└── adapters/
    ├── __init__.py
    ├── providers/
    │   ├── __init__.py
    │   ├── vllm_provider.py      # vLLM HTTP client (PagedAttention)
    │   ├── ollama_provider.py    # Ollama HTTP client
    │   ├── litellm_provider.py   # LiteLLM unified client (100+ providers)
    │   └── openai_provider.py    # Direct OpenAI SDK client
    ├── model_router.py           # Intelligent routing logic
    ├── cost_tracker.py           # tiktoken + cost calculation
    ├── rate_limiter.py           # Redis-based per-tenant rate limiting
    ├── repositories.py           # SQLAlchemy repositories
    └── kafka.py                  # LLM event publishing
```

## Database Conventions

- **Table prefix:** `llm_` (e.g., `llm_requests`, `llm_model_configs`, `llm_tenant_quotas`)
- All tenant-scoped tables extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)

## API Conventions

This service exposes TWO API namespaces:

### OpenAI-Compatible (for drop-in compatibility)
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`

### AumOS Extensions (for platform management)
- `GET /serving/usage/{tenant_id}` — usage dashboard
- `POST /serving/models` — configure model
- `PUT /serving/models/{id}` — update model config
- `POST /serving/quotas` — set tenant quota

## Repo-Specific Context

### LLM Provider Priority
1. **vLLM** — use for production when GPU available; supports PagedAttention, continuous batching
2. **Ollama** — use for local dev, air-gapped deployments, and small models
3. **LiteLLM** — use for cloud providers (OpenAI, Anthropic, Azure, Cohere, etc.)
4. **OpenAI direct** — use when LiteLLM overhead is unacceptable

### Intelligent Routing Logic
Route based on:
- `task_type` hint: `"code"` → GPT-4/Claude, `"embedding"` → text-embedding-3, `"fast"` → GPT-3.5/Haiku
- `max_cost_usd`: route to cheapest capable model
- `max_latency_ms`: route to fastest available model
- Provider health: skip unhealthy providers automatically
- Tenant quota: enforce daily/monthly limits before routing

### Token Counting
- Use `tiktoken` for OpenAI-family models
- Use provider-reported token counts for other providers (via LiteLLM response)
- Always record both prompt_tokens and completion_tokens separately

### Cost Calculation
- Maintain in-memory cost table (updated from LiteLLM model pricing)
- Cost = (prompt_tokens * cost_per_input_token) + (completion_tokens * cost_per_output_token)
- Prices stored per-model in `llm_model_configs` table (admin-configurable)

### Rate Limiting Keys
- Per-tenant RPM: `rate:rpm:{tenant_id}:{minute_bucket}`
- Per-tenant TPM: `rate:tpm:{tenant_id}:{minute_bucket}`
- Use sliding window counters in Redis

### Performance Requirements
- Chat completion latency overhead (routing + tracking): < 5ms
- Token counting: < 1ms per request
- Rate limit check: < 2ms per request
- Target end-to-end latency (excluding LLM inference): < 10ms

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT hardcode model names or provider URLs.** Always use settings/config.
8. **Do NOT log API keys or provider credentials.** Ever.
9. **Do NOT bypass rate limiting or quota checks.** Every request must be checked.
10. **Do NOT mix OpenAI-compatible and AumOS schemas.** Keep them separate in schemas.py.
