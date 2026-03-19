import os

# Define env vars before any app imports so pydantic-settings doesn't fail
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("AI_MODEL", "gpt-4o")
