from fastapi import APIRouter
from app.services import db_client

router = APIRouter()


@router.get("/health")
def health_check():
    try:
        db_client.read("pdf_uploads", {})
        edge_status = "connected"
    except Exception as e:
        edge_status = f"error: {str(e)}"

    return {"status": "ok", "edge_function": edge_status}
