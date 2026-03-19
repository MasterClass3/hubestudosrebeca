import hmac
import json
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from app.config import get_settings
from app.routes.pipeline import _run_pipeline

router = APIRouter()


@router.post("/webhook/pdf-uploaded")
async def pdf_uploaded_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Recebe evento do Supabase/Lovable quando um PDF é inserido em pdf_uploads.
    Verifica o webhook_secret e dispara o pipeline em background.
    """
    settings = get_settings()
    payload = await request.body()

    # Verifica secret se configurado
    if settings.webhook_secret:
        signature = request.headers.get("x-webhook-secret", "")
        if not hmac.compare_digest(signature, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="Assinatura inválida")

    body = json.loads(payload)

    # Supabase Database Webhook payload: { type, table, record, old_record }
    record = body.get("record", {})
    pdf_upload_id = record.get("id")

    if not pdf_upload_id:
        raise HTTPException(status_code=400, detail="Campo 'record.id' não encontrado no payload")

    # Só processa quando status é 'pending'
    if record.get("status") != "pending":
        return {"skipped": True, "reason": "status não é pending"}

    background_tasks.add_task(_run_pipeline, pdf_upload_id)
    return {"status": "processing", "pdf_upload_id": pdf_upload_id}
