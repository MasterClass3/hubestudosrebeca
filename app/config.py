from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    supabase_url: str
    supabase_anon_key: str = ""
    anthropic_api_key: str
    ai_model: str = "claude-sonnet-4-6"
    ai_extraction_model: str = "claude-haiku-4-5-20251001"
    extraction_parallelism: int = 8
    callback_url: str = "https://epdiqyrhfkwfigdcpngw.supabase.co/functions/v1/process-callback"
    webhook_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
