import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.callback_service import get_client
from app.services.pdf_service import download_and_extract_text, PDFExtractionError, PDFScannedError
from app.services.ai_extraction import extract_and_save_questions
from app.services.ai_analysis import generate_analysis_for_questions
from app.services.syllabus_service import extract_and_save_syllabus

logger = logging.getLogger(__name__)

router = APIRouter()


class ProcessRequest(BaseModel):
    pdf_upload_id: str


class GenerateAnalysisRequest(BaseModel):
    question_ids: list[str]


class ExtractSyllabusRequest(BaseModel):
    text: str
    study_plan_id: str


def _run_pipeline(pdf_upload_id: str):
    cb = get_client()
    etapa = "0: inicializando"
    try:
        etapa = "1: buscando pdf_upload do banco"
        logger.info(f"[Pipeline] Etapa {etapa}")
        upload = cb.get_pdf_upload(pdf_upload_id)
        if not upload:
            logger.error(f"[Pipeline] pdf_upload_id '{pdf_upload_id}' não encontrado no banco")
            return

        etapa = "2: marcando como processing"
        logger.info(f"[Pipeline] Etapa {etapa}")
        cb.update_pdf_status(pdf_upload_id, "processing")

        etapa = "3: gerando signed URL e baixando PDF"
        file_path = upload["file_path"]
        logger.info(f"[Pipeline] Etapa {etapa} — file_path='{file_path}'")
        text = download_and_extract_text(file_path)
        logger.info(f"[Pipeline] Texto extraído: {len(text)} caracteres")

        pdf_type = upload.get("type")
        if pdf_type == "exam":
            etapa = "4: extraindo questões com IA"
            logger.info(f"[Pipeline] Etapa {etapa}")
            question_ids = extract_and_save_questions(
                raw_text=text,
                study_plan_id=upload["study_plan_id"],
                source_pdf_id=pdf_upload_id,
            )
            logger.info(f"[Pipeline] Questões extraídas: {len(question_ids)}")

            if question_ids:
                etapa = "5: gerando análises (justificativas/peguinhas)"
                logger.info(f"[Pipeline] Etapa {etapa}")
                generate_analysis_for_questions(question_ids)

        elif pdf_type == "syllabus":
            etapa = "4: extraindo conteúdo programático"
            logger.info(f"[Pipeline] Etapa {etapa}")
            extract_and_save_syllabus(text=text, study_plan_id=upload["study_plan_id"])

        etapa = "6: marcando como completed"
        logger.info(f"[Pipeline] Etapa {etapa}")
        cb.update_pdf_status(pdf_upload_id, "completed")
        logger.info(f"[Pipeline] Pipeline concluído com sucesso para '{pdf_upload_id}'")

    except (PDFScannedError, PDFExtractionError) as e:
        msg = repr(e)
        logger.error(f"[Pipeline] ERRO na etapa {etapa}: {msg}")
        cb.update_pdf_status(pdf_upload_id, "error", error_message=msg)
    except Exception as e:
        msg = repr(e)
        logger.error(f"[Pipeline] ERRO INESPERADO na etapa {etapa}: {msg}")
        cb.update_pdf_status(pdf_upload_id, "error", error_message=msg)


@router.post("/pipeline/process")
def process_pdf(request: ProcessRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_pipeline, request.pdf_upload_id)
    return {"status": "processing", "pdf_upload_id": request.pdf_upload_id}


@router.get("/pipeline/status/{pdf_upload_id}")
def get_pipeline_status(pdf_upload_id: str):
    upload = get_client().get_pdf_upload(pdf_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="pdf_upload_id não encontrado")
    return upload


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
