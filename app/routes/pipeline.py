import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.callback_service import get_client
from app.services.pdf_service import download_and_extract_text, PDFExtractionError, PDFScannedError
from app.services.ai_extraction import extract_and_save_questions
from app.services.ai_analysis import generate_analysis_for_questions
from app.services.syllabus_service import extract_and_save_syllabus

logger = logging.getLogger(__name__)
router = APIRouter()

# Timeout máximo de processamento (10 minutos)
MAX_PROCESSING_SECONDS = 600


class ProcessRequest(BaseModel):
    pdf_upload_id: str


class CancelRequest(BaseModel):
    pdf_upload_id: str


class GenerateAnalysisRequest(BaseModel):
    question_ids: list[str]


class ExtractSyllabusRequest(BaseModel):
    text: str
    study_plan_id: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_cancel(cb, pdf_upload_id: str, stage: str):
    """Lança CancelledError se cancelamento foi solicitado."""
    if cb.check_cancel_requested(pdf_upload_id):
        raise InterruptedError(f"Cancelamento solicitado na etapa: {stage}")


def _check_timeout(started_at: float, pdf_upload_id: str, stage: str):
    """Lança TimeoutError se o job excedeu o tempo máximo."""
    elapsed = time.time() - started_at
    if elapsed > MAX_PROCESSING_SECONDS:
        raise TimeoutError(
            f"Timeout após {int(elapsed)}s na etapa '{stage}'"
        )


def _run_pipeline(pdf_upload_id: str):
    cb = get_client()
    started_at = time.time()
    stage = "Iniciando"

    try:
        # Etapa 1 — busca registro e inicia
        stage = "Preparando o documento"
        logger.info(f"[Pipeline:{pdf_upload_id}] {stage}")
        upload = cb.get_pdf_upload(pdf_upload_id)
        if not upload:
            logger.error(f"[Pipeline:{pdf_upload_id}] Registro não encontrado no banco")
            return

        cb.update_pdf_status(
            pdf_upload_id, "processing",
            progress=5, stage=stage,
            processing_started_at=_now_iso(),
        )
        _check_cancel(cb, pdf_upload_id, stage)

        # Etapa 2 — download
        stage = "Baixando o arquivo"
        logger.info(f"[Pipeline:{pdf_upload_id}] {stage}")
        cb.update_heartbeat(pdf_upload_id, 15, stage)
        _check_cancel(cb, pdf_upload_id, stage)
        _check_timeout(started_at, pdf_upload_id, stage)

        file_path = upload["file_path"]
        logger.info(f"[Pipeline:{pdf_upload_id}] file_path='{file_path}'")

        # Etapa 3 — extração de texto
        stage = "Lendo o conteúdo"
        logger.info(f"[Pipeline:{pdf_upload_id}] {stage}")
        cb.update_heartbeat(pdf_upload_id, 25, stage)
        text = download_and_extract_text(file_path)
        logger.info(f"[Pipeline:{pdf_upload_id}] Texto extraído: {len(text)} chars")
        _check_cancel(cb, pdf_upload_id, stage)
        _check_timeout(started_at, pdf_upload_id, stage)

        pdf_type = upload.get("type")

        if pdf_type == "exam":
            stage = "Identificando as questões"
            logger.info(f"[Pipeline:{pdf_upload_id}] {stage}")
            cb.update_heartbeat(pdf_upload_id, 40, stage)
            _check_cancel(cb, pdf_upload_id, stage)
            _check_timeout(started_at, pdf_upload_id, stage)

            question_ids = extract_and_save_questions(
                raw_text=text,
                study_plan_id=upload["study_plan_id"],
                source_pdf_id=pdf_upload_id,
            )
            logger.info(f"[Pipeline:{pdf_upload_id}] {len(question_ids)} questões extraídas (com justificativas)")

            stage = "Salvando no banco"
            cb.update_heartbeat(pdf_upload_id, 85, stage)

        elif pdf_type == "syllabus":
            stage = "Identificando o conteúdo programático"
            logger.info(f"[Pipeline:{pdf_upload_id}] {stage}")
            cb.update_heartbeat(pdf_upload_id, 40, stage)
            _check_cancel(cb, pdf_upload_id, stage)
            _check_timeout(started_at, pdf_upload_id, stage)

            extract_and_save_syllabus(text=text, study_plan_id=upload["study_plan_id"])
            cb.update_heartbeat(pdf_upload_id, 80, "Salvando no banco")

        # Concluído
        stage = "Finalizando"
        logger.info(f"[Pipeline:{pdf_upload_id}] Concluído com sucesso")
        cb.update_pdf_status(
            pdf_upload_id, "completed",
            progress=100, stage="Concluído",
            completed_at=_now_iso(),
        )

    except InterruptedError as e:
        msg = str(e)
        logger.info(f"[Pipeline:{pdf_upload_id}] CANCELADO — {msg}")
        cb.update_pdf_status(pdf_upload_id, "cancelled", stage="Cancelado pelo usuário", error_message=msg)

    except TimeoutError as e:
        msg = str(e)
        logger.error(f"[Pipeline:{pdf_upload_id}] TIMEOUT — {msg}")
        cb.update_pdf_status(pdf_upload_id, "stalled", stage="Tempo limite excedido", error_message=msg)

    except (PDFScannedError, PDFExtractionError) as e:
        msg = repr(e)
        logger.error(f"[Pipeline:{pdf_upload_id}] ERRO PDF na etapa '{stage}': {msg}")
        cb.update_pdf_status(pdf_upload_id, "error", stage=f"Erro: {stage}", error_message=msg)

    except Exception as e:
        msg = repr(e)
        logger.error(f"[Pipeline:{pdf_upload_id}] ERRO INESPERADO na etapa '{stage}': {msg}")
        cb.update_pdf_status(pdf_upload_id, "error", stage=f"Erro inesperado: {stage}", error_message=msg)


def _format_status(upload: dict) -> dict:
    """Formata o registro do banco no contrato estável para a UI."""
    return {
        "id": upload.get("id"),
        "status": upload.get("status", "pending"),
        "progress": upload.get("progress", 0),
        "stage": upload.get("processing_stage") or "",
        "error_message": upload.get("error_message"),
        "cancel_requested": upload.get("cancel_requested", False),
        "last_heartbeat_at": upload.get("last_heartbeat_at"),
        "completed_at": upload.get("completed_at"),
    }


# ------------------------------------------------------------------ #
# Endpoints                                                             #
# ------------------------------------------------------------------ #

@router.post("/pipeline/process")
def process_pdf(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Inicia o pipeline em background."""
    logger.info(f"[Pipeline:{request.pdf_upload_id}] Requisição recebida")
    background_tasks.add_task(_run_pipeline, request.pdf_upload_id)
    return {"status": "processing", "pdf_upload_id": request.pdf_upload_id}


@router.get("/pipeline/status/{pdf_upload_id}")
def get_pipeline_status(pdf_upload_id: str):
    """Retorna o status formatado para a UI."""
    upload = get_client().get_pdf_upload(pdf_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="pdf_upload_id não encontrado")
    return _format_status(upload)


@router.post("/pipeline/cancel")
def cancel_pipeline(request: CancelRequest):
    """Solicita o cancelamento de um job em andamento."""
    upload = get_client().get_pdf_upload(request.pdf_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="pdf_upload_id não encontrado")
    if upload.get("status") not in ("pending", "processing"):
        raise HTTPException(status_code=400, detail=f"Job não pode ser cancelado — status atual: {upload.get('status')}")
    get_client().request_cancel(request.pdf_upload_id)
    logger.info(f"[Pipeline:{request.pdf_upload_id}] Cancelamento solicitado via API")
    return {"status": "cancel_requested", "pdf_upload_id": request.pdf_upload_id}


@router.post("/generate-analysis")
def generate_analysis(request: GenerateAnalysisRequest):
    if not request.question_ids:
        raise HTTPException(status_code=400, detail="question_ids não pode ser vazio")
    return generate_analysis_for_questions(request.question_ids)


@router.post("/extract-syllabus")
def extract_syllabus(request: ExtractSyllabusRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text não pode ser vazio")
    return extract_and_save_syllabus(text=request.text, study_plan_id=request.study_plan_id)
