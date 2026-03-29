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
        """Retorna o registro completo de pdf_uploads pelo id."""
        result = self.call("read", {"table": "pdf_uploads", "filters": {"id": pdf_upload_id}})
        rows = result.get("data", result) if isinstance(result, dict) else result
        # Protege contra Edge Function devolver dict direto em vez de list
        if isinstance(rows, list):
            return rows[0] if rows else {}
        if isinstance(rows, dict):
            return rows
        return {}

    def get_questions(self, study_plan_id: str) -> list:
        result = self.call("read", {"table": "questions", "filters": {"study_plan_id": study_plan_id}})
        if isinstance(result, dict):
            return result.get("data", [])
        return result

    def read(self, table: str, filters: dict | None = None) -> list:
        data: dict = {"table": table}
        if filters:
            data["filters"] = filters
        result = self.call("read", data)
        if isinstance(result, dict):
            return result.get("data", [])
        return result

    def check_cancel_requested(self, pdf_upload_id: str) -> bool:
        """Verifica se o cancelamento foi solicitado para este job."""
        upload = self.get_pdf_upload(pdf_upload_id)
        return bool(upload.get("cancel_requested", False))

    # ------------------------------------------------------------------ #
    # Escrita de status / progresso                                         #
    # ------------------------------------------------------------------ #

    def update_pdf_status(
        self,
        pdf_id: str,
        status: str,
        *,
        progress: int | None = None,
        stage: str | None = None,
        error_message: str | None = None,
        processing_started_at: str | None = None,
        completed_at: str | None = None,
        questions_count: int | None = None,
    ):
        """
        Muda o status do job (processing, completed, error, cancelled, stalled).
        Usar em transições de estado — não em atualizações intermediárias.
        """
        data: dict = {"pdf_id": pdf_id, "status": status}
        if progress is not None:
            data["progress"] = progress
        if stage is not None:
            data["processing_stage"] = stage
        if error_message is not None:
            data["error_message"] = error_message
        if processing_started_at is not None:
            data["processing_started_at"] = processing_started_at
        if completed_at is not None:
            data["completed_at"] = completed_at
        if questions_count is not None:
            data["questions_count"] = questions_count
        return self.call("update_pdf_status", data)

    def update_heartbeat(self, pdf_id: str, progress: int, stage: str):
        """
        Atualização leve entre etapas — atualiza progress, stage e last_heartbeat_at.
        Não muda o status.
        """
        return self.call("update_heartbeat", {
            "pdf_id": pdf_id,
            "progress": progress,
            "processing_stage": stage,
        })

    def request_cancel(self, pdf_id: str):
        """Marca cancel_requested = true no banco."""
        return self.call("update_pdf_status", {"pdf_id": pdf_id, "cancel_requested": True})

    # ------------------------------------------------------------------ #
    # Inserções                                                             #
    # ------------------------------------------------------------------ #

    def upsert_subject(self, name: str, study_plan_id: str) -> str:
        # 1. Tenta ler subject existente
        rows = self.read("subjects", {"study_plan_id": study_plan_id, "name": name})
        if rows:
            first = rows[0] if isinstance(rows, list) else rows
            subject_id = first.get("id")
            if subject_id:
                return subject_id

        # 2. Insere novo subject
        result = self.call("insert_subjects", {"subjects": [{"study_plan_id": study_plan_id, "name": name}]})
        data = result.get("data", []) if isinstance(result, dict) else []
        if data:
            first = data[0] if isinstance(data, list) else data
            subject_id = (first or {}).get("id")
            if subject_id:
                return subject_id

        # 3. Read-after-write: Edge Function pode não retornar "id" no insert
        #    (Supabase .insert() sem .select() retorna null — buscamos direto)
        rows = self.read("subjects", {"study_plan_id": study_plan_id, "name": name})
        if rows:
            first = rows[0] if isinstance(rows, list) else rows
            subject_id = first.get("id")
            if subject_id:
                return subject_id

        raise RuntimeError(f"Falha ao criar disciplina '{name}' — id não encontrado após insert")

    def insert_questions(self, questions: list, study_plan_id: str, source_pdf_id: str) -> list[str]:
        for q in questions:
            q.setdefault("study_plan_id", study_plan_id)
            q.setdefault("source_pdf_id", source_pdf_id)

        result = self.call("insert_questions", {"questions": questions})

        # Tenta extrair IDs da resposta direta
        ids: list[str] = []
        if isinstance(result, dict):
            # Formato {"ids": [...]}
            ids = result.get("ids") or []
            # Formato {"data": [{"id": ...}, ...]}
            if not ids:
                data = result.get("data") or []
                if isinstance(data, list):
                    ids = [row["id"] for row in data if isinstance(row, dict) and row.get("id")]

        if ids:
            return ids

        # Read-after-write: Edge Function pode não retornar IDs no insert
        # (Supabase .insert() sem .select() retorna null — buscamos direto)
        logger.info(
            f"[insert_questions] insert não retornou IDs — "
            f"buscando por source_pdf_id={source_pdf_id}"
        )
        rows = self.read("questions", {"source_pdf_id": source_pdf_id})
        if rows and isinstance(rows, list):
            ids = [row["id"] for row in rows if isinstance(row, dict) and row.get("id")]

        if ids:
            logger.info(f"[insert_questions] read-after-write encontrou {len(ids)} IDs")
            return ids

        logger.error(
            f"[insert_questions] não foi possível obter IDs após insert "
            f"(source_pdf_id={source_pdf_id})"
        )
        return []

    def insert_justifications(self, justifications: list):
        return self.call("insert_justifications", {"justifications": justifications})

    def insert_tricky_points(self, tricky_points: list):
        return self.call("insert_tricky_points", {"tricky_points": tricky_points})

    def insert_syllabus_topics(self, topics: list, study_plan_id: str):
        for t in topics:
            t.setdefault("study_plan_id", study_plan_id)
        return self.call("insert_syllabus_topics", {"topics": topics})

    # ------------------------------------------------------------------ #
    # Storage                                                               #
    # ------------------------------------------------------------------ #

    def save_text_content(self, pdf_upload_id: str, text: str):
        """Persiste o texto extraído do PDF em pdf_uploads.text_content para reuso."""
        return self.call("save_text_content", {"pdf_id": pdf_upload_id, "text_content": text})

    def update_pdf_concurso_name(self, pdf_id: str, concurso_name: str):
        """
        Salva o nome do concurso (sem a banca) em pdf_uploads.concurso_name.
        Usado pelo smart parser para preencher o nome de exibição.
        """
        return self.call("update_pdf_status", {
            "pdf_id": pdf_id,
            "concurso_name": concurso_name,
        })

    # ------------------------------------------------------------------ #
    # Storage                                                               #
    # ------------------------------------------------------------------ #

    def get_signed_url(self, file_path: str, bucket: str = "pdfs", expires_in: int = 120) -> str:
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
            raise RuntimeError(f"Edge Function retornou {response.status_code}: {response.text[:300]}")

        result = response.json()
        signed_url = (
            (result.get("data") or {}).get("signed_url")
            or (result.get("data") or {}).get("signedUrl")
            or result.get("signed_url")
            or result.get("signedUrl")
            or result.get("signedURL")
            or (result.get("payload") or {}).get("signed_url")
            or (result.get("payload") or {}).get("signedUrl")
        )
        if not signed_url:
            raise RuntimeError(f"signed_url não encontrada na resposta: {response.text[:300]}")
        return signed_url


# Singleton
_client: SupabaseCallbackClient | None = None


def get_client() -> SupabaseCallbackClient:
    global _client
    if _client is None:
        _client = SupabaseCallbackClient()
    return _client
