# Contributing to aumos-llm-serving

Thank you for your interest in contributing to the AumOS LLM Serving service.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/muveraai/aumos-llm-serving.git
cd aumos-llm-serving

# Install dependencies
make install

# Copy environment file
cp .env.example .env

# Start dev services
make docker-run
```

## Code Standards

- Python 3.11+ with strict type hints on all function signatures
- Ruff for linting and formatting (`make lint`, `make format`)
- Mypy strict mode (`make typecheck`)
- 80% test coverage minimum (`make test`)
- Follow hexagonal architecture: api/ → core/ → adapters/

## Commit Convention

Use conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring without behavior change
- `test:` — adding or updating tests
- `docs:` — documentation changes
- `chore:` — maintenance tasks

## Pull Request Process

1. Branch from `main` with prefix: `feature/`, `fix/`, or `docs/`
2. Run `make all` before submitting
3. Write tests for new functionality
4. Update CHANGELOG.md under `[Unreleased]`
5. Request review from at least one maintainer

## Architecture Rules

- **Never bypass aumos-common**: Use its auth, DB, events, logging, and health primitives
- **OpenAI-compatible API**: All LLM endpoints must maintain OpenAI API compatibility
- **Tenant isolation**: All data access must respect RLS via aumos-common
- **Cost tracking**: Every LLM call must record token usage and cost
- **Provider abstraction**: New providers implement `LLMProviderProtocol`
