# Changelog

All notable changes to `aumos-llm-serving` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-llm-serving
- OpenAI-compatible API (chat/completions, completions, embeddings, models)
- vLLM provider adapter for production GPU serving with PagedAttention
- Ollama provider adapter for development and small deployments
- LiteLLM provider adapter for 100+ provider routing
- OpenAI direct provider adapter
- Intelligent model router (task type, cost, latency-aware)
- Per-tenant cost tracking with tiktoken token counting
- Redis-based rate limiting with quota enforcement
- Model configuration CRUD with health checks and failover
- Tenant quota management with budget enforcement
- Hexagonal architecture (api/ + core/ + adapters/)
- SQLAlchemy models: LLMRequest, ModelConfig, TenantQuota
- Kafka event publishing for LLM lifecycle events
- Full standard deliverables (Docker, CI, docs)
