from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services.callback_service import get_client
from app.services.pdf_service import download_and_extract_text, PDFExtractionError, PDFScannedError
from app.services.ai_extraction import extract_and_save_questions
from app.services.ai_analysis import generate_analysis_for_questions
from app.services.syllabus_service import extract_and_save_syllabus

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
    try:
        upload = cb.get_pdf_upload(pdf_upload_id)
        if not upload:
            return

        cb.update_pdf_status(pdf_upload_id, "processing")
        text = download_and_extract_text(upload["file_path"])

        if upload["type"] == "exam":
            question_ids = extract_and_save_questions(
                raw_text=text,
                study_plan_id=upload["study_plan_id"],
                source_pdf_id=pdf_upload_id,
            )
            if question_ids:
                generate_analysis_for_questions(question_ids)

        elif upload["type"] == "syllabus":
            extract_and_save_syllabus(text=text, study_plan_id=upload["study_plan_id"])

        cb.update_pdf_status(pdf_upload_id, "completed")

    except PDFScannedError as e:
        get_client().update_pdf_status(pdf_upload_id, "error", str(e))
    except PDFExtractionError as e:
        get_client().update_pdf_status(pdf_upload_id, "error", str(e))
    except Exception as e:
        get_client().update_pdf_status(pdf_upload_id, "error", f"Erro interno: {e}")


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
