# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not** open a public GitHub issue for security vulnerabilities.

Report security issues to: security@muveraai.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

We aim to respond within 48 hours and provide a fix within 7 days for critical issues.

## Security Considerations for LLM Serving

### API Key Management
- Never log API keys or model credentials
- Store provider API keys in aumos-secrets-vault (OpenBao/Vault)
- Rotate keys regularly via the model management API

### Prompt Injection
- Validate and sanitize all user inputs before forwarding to LLM providers
- Log prompt injection attempts via the security event pipeline
- Use system prompt guardrails for sensitive deployments

### Tenant Isolation
- All LLM requests are scoped to the requesting tenant
- Cost and usage data is RLS-protected per tenant
- Model configurations are tenant-scoped (no cross-tenant model sharing by default)

### Rate Limiting
- Per-tenant rate limits enforced via Redis
- Budget enforcement prevents runaway costs
- Quota alerts sent via Kafka events

### Data Residency
- Configure provider routing to respect data residency requirements
- Use on-premises vLLM or Ollama for sensitive data
- Avoid sending PII to external providers without explicit tenant consent
