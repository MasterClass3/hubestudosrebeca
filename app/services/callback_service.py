"""
SupabaseCallbackClient — todas as operações de banco passam pela
Edge Function process-callback do Supabase.
"""
import logging
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


class SupabaseCallbackClient:

    def __init__(self):
        settings = get_settings()
        self._url = settings.callback_url
        self._secret = settings.webhook_secret

    # ------------------------------------------------------------------ #
    # Método genérico                                                       #
    # ------------------------------------------------------------------ #

    def call(self, action: str, data: dict) -> dict:
        response = httpx.post(
            self._url,
            json={"action": action, "data": data},
            headers={
                "Content-Type": "application/json",
                "x-webhook-secret": self._secret,
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Leitura                                                               #
    # ------------------------------------------------------------------ #

    def get_pdf_upload(self, pdf_upload_id: str) -> dict:
        """Retorna o registro de pdf_uploads pelo id."""
        result = self.call("read", {"table": "pdf_uploads", "filters": {"id": pdf_upload_id}})
        rows = result.get("data", result) if isinstance(result, dict) else result
        return rows[0] if rows else {}

    def get_questions(self, study_plan_id: str) -> list:
        """Retorna todas as questões de um plano."""
        result = self.call("read", {"table": "questions", "filters": {"study_plan_id": study_plan_id}})
        if isinstance(result, dict):
            return result.get("data", [])
        return result

    def read(self, table: str, filters: dict | None = None) -> list:
        """Leitura genérica."""
        data: dict = {"table": table}
        if filters:
            data["filters"] = filters
        result = self.call("read", data)
        if isinstance(result, dict):
            return result.get("data", [])
        return result

    # ------------------------------------------------------------------ #
    # Escrita                                                               #
    # ------------------------------------------------------------------ #

    def update_pdf_status(self, pdf_id: str, status: str, error_message: str | None = None):
        data: dict = {"pdf_id": pdf_id, "status": status}
        if error_message:
            data["error_message"] = error_message
        return self.call("update_pdf_status", data)

    def upsert_subject(self, name: str, study_plan_id: str) -> str:
        """Busca ou cria uma disciplina. Retorna o subject_id."""
        # Tenta ler primeiro
        rows = self.read("subjects", {"study_plan_id": study_plan_id, "name": name})
        if rows:
            return rows[0]["id"]
        # Cria nova
        result = self.call("insert_subjects", {"subjects": [{"study_plan_id": study_plan_id, "name": name}]})
        data = result.get("data", []) if isinstance(result, dict) else []
        if data:
            return data[0]["id"]
        raise RuntimeError(f"Falha ao criar disciplina '{name}'")

    def insert_questions(self, questions: list, study_plan_id: str, source_pdf_id: str) -> list[str]:
        """Insere questões (já enriquecidas com study_plan_id/source_pdf_id) e retorna IDs."""
        for q in questions:
            q.setdefault("study_plan_id", study_plan_id)
            q.setdefault("source_pdf_id", source_pdf_id)
        result = self.call("insert_questions", {"questions": questions})
        if isinstance(result, dict):
            return result.get("ids", [])
        return []

    def insert_justifications(self, justifications: list):
        return self.call("insert_justifications", {"justifications": justifications})

    def insert_tricky_points(self, tricky_points: list):
        return self.call("insert_tricky_points", {"tricky_points": tricky_points})

    def insert_syllabus_topics(self, topics: list, study_plan_id: str):
        for t in topics:
            t.setdefault("study_plan_id", study_plan_id)
        return self.call("insert_syllabus_topics", {"topics": topics})

    def get_signed_url(self, file_path: str, bucket: str = "pdfs", expires_in: int = 60) -> str:
        """Gera uma signed URL para download de arquivo do Supabase Storage."""
        body = {"action": "get_signed_url", "data": {"file_path": file_path, "bucket": bucket, "expires_in": expires_in}}
        logger.info(f"[SignedURL] Enviando: {body}")

        response = httpx.post(
            self._url,
            json=body,
            headers={"Content-Type": "application/json", "x-webhook-secret": self._secret},
            timeout=_TIMEOUT,
        )
        logger.info(f"[SignedURL] status={response.status_code} body={response.text[:500]}")

        if response.status_code != 200:
            raise RuntimeError(
                f"Edge Function retornou {response.status_code}: {response.text[:300]}"
            )

        result = response.json()

        # Trata todas as variações possíveis de resposta
        signed_url = (
            result.get("signed_url")
            or result.get("signedUrl")
            or result.get("signedURL")
            or (result.get("payload") or {}).get("signed_url")
            or (result.get("payload") or {}).get("signedUrl")
            or (result.get("data") or {}).get("signed_url")
            or (result.get("data") or {}).get("signedUrl")
        )

        if not signed_url:
            raise RuntimeError(f"signed_url não encontrada na resposta: {response.text[:300]}")
        return signed_url


# Instância singleton para reutilizar conexão HTTP
_client: SupabaseCallbackClient | None = None


def get_client() -> SupabaseCallbackClient:
    global _client
    if _client is None:
        _client = SupabaseCallbackClient()
    return _client
