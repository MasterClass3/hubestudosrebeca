import io
import httpx
import pdfplumber

from app.config import get_settings

SUPABASE_STORAGE_URL = "https://epdiqyrhfkwfigdcpngw.supabase.co/storage/v1/object"


class PDFExtractionError(Exception):
    pass


class PDFScannedError(PDFExtractionError):
    pass


def download_and_extract_text(file_path: str) -> str:
    """
    Baixa um PDF do Supabase Storage (bucket 'pdfs') e extrai o texto.
    Usa a anon key via REST API do storage.
    """
    settings = get_settings()

    # Tenta download via Storage REST API com a anon key
    url = f"{SUPABASE_STORAGE_URL}/pdfs/{file_path}"
    try:
        response = httpx.get(
            url,
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {settings.supabase_anon_key}",
            },
            timeout=60.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        pdf_bytes = response.content
    except httpx.HTTPStatusError as e:
        raise PDFExtractionError(f"Falha ao baixar PDF '{file_path}': HTTP {e.response.status_code}")
    except Exception as e:
        raise PDFExtractionError(f"Falha ao baixar PDF '{file_path}': {e}")

    if not pdf_bytes:
        raise PDFExtractionError(f"Download retornou vazio para '{file_path}'")

    # Extração de texto
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

    return full_text
