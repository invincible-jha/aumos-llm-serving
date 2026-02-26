# aumos-llm-serving

Unified LLM serving layer for AumOS Enterprise. Provides an OpenAI-compatible API with intelligent multi-provider routing, per-tenant cost tracking, and budget enforcement.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
- **vLLM** — production GPU serving with PagedAttention and continuous batching
- **Ollama** — development and small-scale deployments
- **LiteLLM** — unified routing to 100+ providers (OpenAI, Anthropic, Azure, Cohere, etc.)
- **Intelligent routing** — route by task type, cost constraint, latency target, or provider health
- **Per-tenant cost tracking** — tiktoken-based token counting, cost attribution per model/provider
- **Budget enforcement** — daily/monthly token limits and cost caps per tenant
- **Redis rate limiting** — per-tenant RPM/TPM enforcement with quota management
- **Model management** — CRUD for model configs, provider health checks, failover

## Architecture

```
aumos-platform-core → aumos-auth-gateway → aumos-llm-serving
                                         ↑ (depends on)
aumos-agent-framework ────────────────────┘
aumos-*-engine ───────────────────────────┘
```

### Service Layers

```
api/          FastAPI routes (OpenAI-compatible + AumOS extensions)
  router.py   Endpoint definitions
  schemas.py  OpenAI-compatible + AumOS Pydantic models

core/         Business logic (framework-independent)
  models.py   SQLAlchemy: LLMRequest, ModelConfig, TenantQuota
  services.py ServingService, RoutingService, CostTrackingService,
              RateLimitingService, ModelManagementService
  interfaces.py  Protocols: LLMProviderProtocol, ModelRouterProtocol, etc.

adapters/     External integrations
  providers/
    vllm_provider.py      vLLM HTTP client
    ollama_provider.py    Ollama HTTP client
    litellm_provider.py   LiteLLM unified client
    openai_provider.py    Direct OpenAI client
  model_router.py         Routing logic
  cost_tracker.py         tiktoken + cost calculation
  rate_limiter.py         Redis-based rate limiting
  repositories.py         SQLAlchemy repositories
  kafka.py                Event publishing
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your provider credentials

# Start services
docker compose -f docker-compose.dev.yml up -d

# Run migrations
make migrate

# Start server
uvicorn aumos_llm_serving.main:app --reload
```

## API Reference

### Chat Completions (OpenAI-compatible)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Usage Dashboard

```bash
curl http://localhost:8000/serving/usage/$TENANT_ID \
  -H "Authorization: Bearer $TOKEN"
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AUMOS_LLM__VLLM_BASE_URL` | vLLM server URL | `http://localhost:8080` |
| `AUMOS_LLM__OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `AUMOS_LLM__LITELLM_CONFIG_PATH` | LiteLLM YAML config | `/app/config/litellm_config.yaml` |
| `AUMOS_LLM__DEFAULT_MODEL` | Default model ID | `ollama/llama3.2` |
| `AUMOS_LLM__ENABLE_COST_TRACKING` | Enable cost tracking | `true` |
| `AUMOS_LLM__DEFAULT_RPM_LIMIT` | Default requests/minute | `60` |
| `AUMOS_LLM__DEFAULT_TPM_LIMIT` | Default tokens/minute | `100000` |

## Development

```bash
make lint       # Ruff linting
make format     # Ruff formatting
make typecheck  # Mypy strict
make test       # Pytest with coverage
make all        # All of the above
```

## Release Tier

**Tier B — Open Core**. Core serving infrastructure is open. Advanced routing strategies and enterprise cost analytics are proprietary.
