import io
import logging
import httpx
import pdfplumber

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    pass


class PDFScannedError(PDFExtractionError):
    pass


def download_and_extract_text(file_path: str) -> str:
    """
    1. Obtém uma signed URL via Edge Function (process-callback)
    2. Baixa o PDF com httpx usando a signed URL
    3. Extrai o texto com pdfplumber
    """
    from app.services.callback_service import get_client

    logger.info(f"[PDF] Solicitando signed URL para file_path='{file_path}' bucket='pdfs'")

    cb = get_client()
    try:
        signed_url = cb.get_signed_url(file_path, bucket="pdfs", expires_in=120)
        logger.info(f"[PDF] Signed URL obtida: {signed_url[:80]}...")
    except Exception as e:
        raise PDFExtractionError(f"Falha ao obter signed URL para '{file_path}': {e}")

    logger.info(f"[PDF] Baixando PDF via signed URL")
    try:
        response = httpx.get(signed_url, timeout=120.0, follow_redirects=True)
        logger.info(f"[PDF] Download HTTP status: {response.status_code}")
        response.raise_for_status()
        pdf_bytes = response.content
    except httpx.HTTPStatusError as e:
        raise PDFExtractionError(
            f"Falha ao baixar PDF '{file_path}': HTTP {e.response.status_code} — {e.response.text[:200]}"
        )
    except Exception as e:
        raise PDFExtractionError(f"Falha ao baixar PDF '{file_path}': {e}")

    if not pdf_bytes:
        raise PDFExtractionError(f"Download retornou vazio para '{file_path}'")

    logger.info(f"[PDF] PDF baixado com sucesso ({len(pdf_bytes)} bytes). Extraindo texto...")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                raise PDFExtractionError("PDF sem páginas.")

            pages_text: list[str] = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            full_text = "\n\n".join(pages_text).strip()
    except PDFExtractionError:
        raise
    except Exception as e:
        raise PDFExtractionError(f"Erro ao processar PDF: {e}")

    if not full_text:
        raise PDFScannedError(
            "Nenhum texto extraível encontrado. "
            "O PDF parece ser escaneado (imagem). OCR não é suportado por enquanto."
        )

    logger.info(f"[PDF] Texto extraído com sucesso ({len(full_text)} caracteres)")
    return full_text
