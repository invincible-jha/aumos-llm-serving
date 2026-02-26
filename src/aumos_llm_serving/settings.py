"""Service-specific settings extending AumOS base config."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class LLMSettings(AumOSSettings):
    """Settings for the aumos-llm-serving service.

    All settings can be overridden via environment variables with the
    AUMOS_LLM__ prefix (e.g., AUMOS_LLM__DEFAULT_MODEL).
    """

    service_name: str = "aumos-llm-serving"

    # vLLM (production GPU serving)
    vllm_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the vLLM HTTP server",
    )
    vllm_api_key: str = Field(
        default="",
        description="API key for vLLM server (if authentication is enabled)",
    )

    # Ollama (development / small deployments)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server",
    )

    # LiteLLM config
    litellm_config_path: str = Field(
        default="/app/config/litellm_config.yaml",
        description="Path to LiteLLM YAML configuration file",
    )

    # Default model
    default_model: str = Field(
        default="ollama/llama3.2",
        description="Default model identifier to use when none is specified",
    )

    # Cost tracking
    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable per-tenant token consumption and cost tracking",
    )
    cost_alert_threshold_usd: float = Field(
        default=100.0,
        description="USD cost threshold to trigger an alert event",
    )

    # Rate limiting
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Sliding window size in seconds for rate limiting",
    )
    default_rpm_limit: int = Field(
        default=60,
        description="Default requests-per-minute limit per tenant",
    )
    default_tpm_limit: int = Field(
        default=100_000,
        description="Default tokens-per-minute limit per tenant",
    )

    # HTTP client settings for provider adapters
    provider_timeout_seconds: float = Field(
        default=120.0,
        description="HTTP timeout for LLM provider requests in seconds",
    )
    provider_max_retries: int = Field(
        default=2,
        description="Maximum number of retries for failed provider requests",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_LLM__")
