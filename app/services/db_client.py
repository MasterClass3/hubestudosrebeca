"""
Cliente para a Edge Function do Supabase (process-callback).
Todas as operações de banco passam por aqui.
"""
import httpx
from app.config import get_settings

EDGE_FUNCTION_URL = "https://epdiqyrhfkwfigdcpngw.supabase.co/functions/v1/process-callback"
_TIMEOUT = 30.0


def _call(action: str, data: dict) -> dict:
    settings = get_settings()
    response = httpx.post(
        EDGE_FUNCTION_URL,
        json={"action": action, "data": data},
        headers={
            "Content-Type": "application/json",
            "x-webhook-secret": settings.webhook_secret,
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def read(table: str, filters: dict | None = None) -> list:
    data: dict = {"table": table}
    if filters:
        data["filters"] = filters
    result = _call("read", data)
    # Aceita { data: [...] } ou lista direta
    if isinstance(result, list):
        return result
    return result.get("data", [])


def update_pdf_status(pdf_id: str, status: str, error_message: str | None = None):
    data: dict = {"pdf_id": pdf_id, "status": status}
    if error_message:
        data["error_message"] = error_message
    return _call("update_pdf_status", data)


def insert_questions(questions: list) -> list[str]:
    """Insere questões e retorna lista de IDs criados."""
    result = _call("insert_questions", {"questions": questions})
    if isinstance(result, dict):
        return result.get("ids", [])
    return []


def insert_subjects(subjects: list) -> list[dict]:
    """Insere disciplinas e retorna lista com id e name."""
    result = _call("insert_subjects", {"subjects": subjects})
    if isinstance(result, dict):
        return result.get("data", [])
    return []


def insert_justifications(justifications: list):
    return _call("insert_justifications", {"justifications": justifications})


def insert_tricky_points(tricky_points: list):
    return _call("insert_tricky_points", {"tricky_points": tricky_points})


def insert_syllabus_topics(topics: list):
    return _call("insert_syllabus_topics", {"topics": topics})
