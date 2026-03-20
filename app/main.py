import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.routes import health, pipeline, questions, webhook

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="HubEstudos Backend",
    description="Backend para processamento de PDFs de concursos com IA",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])
app.include_router(questions.router, prefix="/api", tags=["Questions"])
app.include_router(webhook.router, prefix="/api", tags=["Webhook"])
