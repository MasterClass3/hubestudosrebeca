from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    supabase_url: str
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    anthropic_api_key: str
    ai_model: str = "claude-sonnet-4-6"
    webhook_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
