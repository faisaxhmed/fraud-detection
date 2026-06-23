"""Loads runtime configuration from environment variables / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized app config; nothing here is hardcoded elsewhere."""

    model_dir: str = "models"
    allowed_origins: str = "http://localhost:3000"
    rate_limit: str = "20/minute"
    narrative_rate_limit: str = "5/minute"
    review_rate_limit: str = "3/minute"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_timeout_seconds: float = 30.0
    anthropic_max_tokens: int = 500
    agent_max_iterations: int = 5
    agent_tool_timeout_seconds: float = 20.0
    agent_max_tokens: int = 1024

    model_config = SettingsConfigDict(env_file=".env", protected_namespaces=())

    @property
    def allowed_origins_list(self) -> list[str]:
        """Splits the comma-separated ALLOWED_ORIGINS env var into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
