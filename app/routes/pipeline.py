import logging
import time
import traceback
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.callback_service import get_client
from app.services.pdf_service import download_and_extract_text, PDFExtractionError, PDFScannedError
from app.services.ai_extraction import extract_and_save_questions, save_parsed_questions
from app.services.syllabus_service import extract_and_save_syllabus
from app.services.smart_parser import is_structured_exam, parse_structured_exam, _extract_meta

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PROCESSING_SECONDS = 600      # 10 min timeout geral
STALE_HEARTBEAT_MINUTES = 10      # job sem heartbeat por 10 min → stalled


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


def _log(pdf_id: str, event: str, msg: str):
    logger.info(f"[{event}] [{pdf_id}] {datetime.now(timezone.utc).isoformat()} — {msg}")


# ------------------------------------------------------------------ #
# Worker principal                                                      #
# ------------------------------------------------------------------ #

def _run_pipeline(pdf_upload_id: str):
    cb = get_client()
    started_at = time.time()
    current_stage = "Iniciando"

    def heartbeat(progress: int, stage: str):
        nonlocal current_stage
        current_stage = stage
        _log(pdf_upload_id, "HEARTBEAT", f"{stage} ({progress}%)")
        try:
            cb.update_heartbeat(pdf_upload_id, progress, stage)
        except Exception as hb_err:
            logger.warning(f"[HEARTBEAT] [{pdf_upload_id}] falhou: {hb_err}")

    def check_cancel():
        if cb.check_cancel_requested(pdf_upload_id):
            raise InterruptedError(f"Cancelamento solicitado na etapa: {current_stage}")

    def check_timeout():
        elapsed = time.time() - started_at
        if elapsed > MAX_PROCESSING_SECONDS:
            raise TimeoutError(f"Timeout ({int(elapsed)}s) na etapa: {current_stage}")

    try:
        # ── ETAPA 1: buscar registro ─────────────────────────────────────
        current_stage = "Preparando o documento"
        _log(pdf_upload_id, "START", current_stage)
        upload = cb.get_pdf_upload(pdf_upload_id)
        if not upload:
            logger.error(f"[START] [{pdf_upload_id}] registro não encontrado no banco — abortando")
            return

        cb.update_pdf_status(
            pdf_upload_id, "processing",
            progress=5, stage=current_stage,
            processing_started_at=_now_iso(),
        )
        check_cancel()

        # ── ETAPA 2: gerar signed URL ────────────────────────────────────
        current_stage = "Baixando o arquivo"
        _log(pdf_upload_id, "STEP_BEGIN", current_stage)
        heartbeat(15, current_stage)
        check_cancel()
        check_timeout()

        file_path = upload.get("file_path", "")
        _log(pdf_upload_id, "STEP_BEGIN", f"file_path='{file_path}'")

        # ── ETAPA 3: baixar PDF e extrair texto (com cache) ──────────────
        current_stage = "Lendo o conteúdo"
        _log(pdf_upload_id, "STEP_BEGIN", current_stage)
        heartbeat(25, current_stage)

        cached_text = upload.get("text_content", "")
        if cached_text and len(cached_text) > 100:
            text = cached_text
            _log(pdf_upload_id, "STEP_SUCCESS", f"texto em cache: {len(text)} chars — download ignorado")
        else:
            try:
                text = download_and_extract_text(file_path)
            except PDFScannedError as e:
                raise PDFScannedError(e) from e
            except PDFExtractionError as e:
                raise PDFExtractionError(e) from e
            except Exception as e:
                raise RuntimeError(f"Falha ao ler PDF: {repr(e)}") from e

            _log(pdf_upload_id, "STEP_SUCCESS", f"texto extraído: {len(text)} chars")

            # Persiste para reprocessamentos futuros (best-effort)
            try:
                cb.save_text_content(pdf_upload_id, text)
            except Exception as cache_err:
                logger.warning(f"[{pdf_upload_id}] save_text_content falhou (não bloqueante): {cache_err}")
        check_cancel()
        check_timeout()

        pdf_type = upload.get("type")

        if pdf_type == "exam":
            # ── ETAPA 4: identificar questões ────────────────────────────
            current_stage = "Identificando as questões"
            _log(pdf_upload_id, "STEP_BEGIN", current_stage)
            heartbeat(35, current_stage)
            check_cancel()
            check_timeout()

            # ── Diagnóstico: amostra do texto e detecção de formato ──────
            _log(pdf_upload_id, "DIAG", f"Tamanho total do texto: {len(text)} chars")
            _log(pdf_upload_id, "DIAG", f"Amostra (500 chars): {repr(text[:500])}")
            _structured = is_structured_exam(text)
            _log(pdf_upload_id, "DIAG", f"is_structured_exam={_structured}")

            try:
                if _structured:
                    # ── Caminho rápido: smart parser (extração por regex) ─
                    heartbeat(36, "Prova estruturada detectada — extraindo questões")
                    _log(pdf_upload_id, "STEP_BEGIN", "smart_parser")

                    meta, parsed_qs = parse_structured_exam(text)
                    _log(pdf_upload_id, "DIAG", f"smart_parser extraiu {len(parsed_qs)} questões")

                    if meta.concurso_name:
                        _log(pdf_upload_id, "META", f"concurso='{meta.concurso_name}'")
                        heartbeat(37, f"Concurso: {meta.concurso_name[:70]}")
                        try:
                            cb.update_pdf_concurso_name(pdf_upload_id, meta.concurso_name)
                        except Exception as meta_err:
                            logger.warning(
                                f"[{pdf_upload_id}] update_pdf_concurso_name falhou "
                                f"(não bloqueante): {meta_err}"
                            )

                    if not parsed_qs:
                        _log(pdf_upload_id, "WARN", "Smart parser não extraiu questões — usando IA")
                        heartbeat(38, "Usando IA para extração")
                        question_ids = extract_and_save_questions(
                            raw_text=text,
                            study_plan_id=upload["study_plan_id"],
                            source_pdf_id=pdf_upload_id,
                            pdf_upload_id=pdf_upload_id,
                            heartbeat_fn=heartbeat,
                        )
                    else:
                        heartbeat(38, f"{len(parsed_qs)} questões identificadas")
                        question_ids = save_parsed_questions(
                            parsed_questions=parsed_qs,
                            study_plan_id=upload["study_plan_id"],
                            source_pdf_id=pdf_upload_id,
                            pdf_upload_id=pdf_upload_id,
                            heartbeat_fn=heartbeat,
                        )
                else:
                    # ── Caminho padrão: extração via IA ──────────────────
                    # Tenta extrair metadados mesmo sem estrutura detectada
                    # (cabeçalhos "Ano:", "Banca:", "Órgão:", "Prova:" podem estar presentes)
                    try:
                        ai_meta = _extract_meta(text)
                        if ai_meta.concurso_name and ai_meta.concurso_name != "Concurso Importado":
                            _log(pdf_upload_id, "META", f"concurso (IA path)='{ai_meta.concurso_name}'")
                            cb.update_pdf_concurso_name(pdf_upload_id, ai_meta.concurso_name)
                    except Exception as meta_err:
                        logger.warning(f"[{pdf_upload_id}] _extract_meta (IA path) falhou (não bloqueante): {meta_err}")

                    question_ids = extract_and_save_questions(
                        raw_text=text,
                        study_plan_id=upload["study_plan_id"],
                        source_pdf_id=pdf_upload_id,
                        pdf_upload_id=pdf_upload_id,
                        heartbeat_fn=heartbeat,
                    )

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[{pdf_upload_id}] TRACEBACK extração:\n{tb}")
                _log(pdf_upload_id, "DIAG", f"Texto nas primeiras 1000 chars: {repr(text[:1000])}")
                raise RuntimeError(f"Falha na extração de questões: {repr(e)}") from e

            _log(pdf_upload_id, "STEP_SUCCESS", f"{len(question_ids)} questões salvas")
            try:
                cb.update_pdf_status(
                    pdf_upload_id, "processing",
                    questions_count=len(question_ids),
                )
            except Exception:
                pass
            heartbeat(90, "Finalizando")
            check_cancel()

        elif pdf_type == "syllabus":
            current_stage = "Identificando o conteúdo programático"
            _log(pdf_upload_id, "STEP_BEGIN", current_stage)
            heartbeat(40, current_stage)
            check_cancel()
            check_timeout()

            try:
                extract_and_save_syllabus(text=text, study_plan_id=upload["study_plan_id"])
            except Exception as e:
                raise RuntimeError(f"Falha na extração do edital: {repr(e)}") from e

            heartbeat(88, "Salvando tópicos")

        else:
            raise ValueError(f"Tipo de PDF desconhecido: '{pdf_type}'")

        # ── CONCLUSÃO ────────────────────────────────────────────────────
        current_stage = "Concluído"
        _log(pdf_upload_id, "FINISH", "pipeline concluído com sucesso")
        cb.update_pdf_status(
            pdf_upload_id, "completed",
            progress=100, stage="Concluído",
            completed_at=_now_iso(),
        )

    except InterruptedError as e:
        msg = str(e)
        _log(pdf_upload_id, "FINISH", f"CANCELADO — {msg}")
        try:
            cb.update_pdf_status(pdf_upload_id, "cancelled", stage="Cancelado pelo usuário", error_message=msg)
        except Exception:
            pass

    except TimeoutError as e:
        msg = str(e)
        _log(pdf_upload_id, "FINISH", f"TIMEOUT — {msg}")
        try:
            cb.update_pdf_status(pdf_upload_id, "stalled", stage="Tempo limite excedido", error_message=msg)
        except Exception:
            pass

    except (PDFScannedError, PDFExtractionError) as e:
        msg = repr(e)
        _log(pdf_upload_id, "STEP_ERROR", f"etapa '{current_stage}': {msg}")
        try:
            cb.update_pdf_status(pdf_upload_id, "error", stage=f"Erro: {current_stage}", error_message=msg)
        except Exception:
            pass

    except Exception as e:
        msg = repr(e)
        _log(pdf_upload_id, "STEP_ERROR", f"INESPERADO na etapa '{current_stage}': {msg}")
        try:
            cb.update_pdf_status(pdf_upload_id, "error", stage=f"Erro inesperado: {current_stage}", error_message=msg)
        except Exception:
            pass


# ------------------------------------------------------------------ #
# Helpers de resposta                                                   #
# ------------------------------------------------------------------ #

def _detect_stale(upload: dict, pdf_upload_id: str) -> dict:
    """Se o job está em processing mas sem heartbeat por STALE_HEARTBEAT_MINUTES, marca stalled."""
    if upload.get("status") != "processing":
        return upload

    last_hb = upload.get("last_heartbeat_at")
    if not last_hb:
        return upload

    try:
        hb_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - hb_time
        if age > timedelta(minutes=STALE_HEARTBEAT_MINUTES):
            stage = upload.get("processing_stage", "desconhecida")
            msg = f"Heartbeat expirou há {int(age.total_seconds())}s na etapa: {stage}"
            _log(pdf_upload_id, "STALE", msg)
            try:
                get_client().update_pdf_status(pdf_upload_id, "stalled", stage="Travado", error_message=msg)
            except Exception:
                pass
            upload = dict(upload)
            upload["status"] = "stalled"
            upload["error_message"] = msg
    except Exception:
        pass

    return upload


def _format_status(upload: dict) -> dict:
    is_completed = upload.get("status") == "completed"
    study_plan_id = upload.get("study_plan_id")
    return {
        "id": upload.get("id"),
        "status": upload.get("status", "pending"),
        "progress": upload.get("progress", 0),
        "stage": upload.get("processing_stage") or "",
        "error_message": upload.get("error_message"),
        "cancel_requested": upload.get("cancel_requested", False),
        "last_heartbeat_at": upload.get("last_heartbeat_at"),
        "completed_at": upload.get("completed_at"),
        # Metadados do concurso (preenchidos pelo smart parser)
        "concurso_name": upload.get("concurso_name"),
        "questions_count": upload.get("questions_count"),
        # Indica para onde o frontend deve redirecionar ao completar
        "redirect_to": f"/plans/{study_plan_id}" if is_completed and study_plan_id else None,
    }


# ------------------------------------------------------------------ #
# Endpoints                                                             #
# ------------------------------------------------------------------ #

@router.post("/pipeline/process")
def process_pdf(request: ProcessRequest, background_tasks: BackgroundTasks):
    _log(request.pdf_upload_id, "START", "requisição recebida")
    background_tasks.add_task(_run_pipeline, request.pdf_upload_id)
    return {"status": "processing", "pdf_upload_id": request.pdf_upload_id}


@router.get("/pipeline/status/{pdf_upload_id}")
def get_pipeline_status(pdf_upload_id: str):
    upload = get_client().get_pdf_upload(pdf_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="pdf_upload_id não encontrado")
    upload = _detect_stale(upload, pdf_upload_id)
    return _format_status(upload)


@router.post("/pipeline/cancel")
def cancel_pipeline(request: CancelRequest):
    upload = get_client().get_pdf_upload(request.pdf_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="pdf_upload_id não encontrado")
    if upload.get("status") not in ("pending", "processing"):
        raise HTTPException(
            status_code=400,
            detail=f"Job não pode ser cancelado — status atual: {upload.get('status')}"
        )
    get_client().request_cancel(request.pdf_upload_id)
    _log(request.pdf_upload_id, "CANCEL", "cancelamento solicitado via API")
    return {"status": "cancel_requested", "pdf_upload_id": request.pdf_upload_id}


@router.post("/generate-analysis")
def generate_analysis(request: GenerateAnalysisRequest):
    from app.services.ai_analysis import generate_analysis_for_questions
    if not request.question_ids:
        raise HTTPException(status_code=400, detail="question_ids não pode ser vazio")
    return generate_analysis_for_questions(request.question_ids)


@router.post("/extract-syllabus")
def extract_syllabus(request: ExtractSyllabusRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text não pode ser vazio")
    return extract_and_save_syllabus(text=request.text, study_plan_id=request.study_plan_id)
