import io
import pytest
from unittest.mock import MagicMock, patch
import pdfplumber

from app.services.pdf_service import download_and_extract_text, PDFExtractionError, PDFScannedError


def _make_pdf_bytes_with_text(text: str) -> bytes:
    """Cria um PDF mínimo em memória com texto (usando pdfplumber não é possível criar,
    então mockamos o pdfplumber.open diretamente nos testes)."""
    return b"%PDF-1.4 fake"


@patch("app.services.pdf_service.get_supabase_client")
def test_download_failure(mock_client):
    mock_client.return_value.storage.from_.return_value.download.side_effect = Exception("404")
    with pytest.raises(PDFExtractionError, match="Falha ao baixar PDF"):
        download_and_extract_text("nonexistent/file.pdf")


@patch("app.services.pdf_service.get_supabase_client")
def test_empty_download(mock_client):
    mock_client.return_value.storage.from_.return_value.download.return_value = b""
    with pytest.raises(PDFExtractionError, match="Download retornou vazio"):
        download_and_extract_text("empty/file.pdf")


@patch("app.services.pdf_service.pdfplumber")
@patch("app.services.pdf_service.get_supabase_client")
def test_scanned_pdf_raises_error(mock_client, mock_pdfplumber):
    mock_client.return_value.storage.from_.return_value.download.return_value = b"%PDF fake"

    mock_page = MagicMock()
    mock_page.extract_text.return_value = None
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdfplumber.open.return_value = mock_pdf

    with pytest.raises(PDFScannedError, match="OCR não é suportado"):
        download_and_extract_text("scanned/file.pdf")


@patch("app.services.pdf_service.pdfplumber")
@patch("app.services.pdf_service.get_supabase_client")
def test_successful_extraction(mock_client, mock_pdfplumber):
    mock_client.return_value.storage.from_.return_value.download.return_value = b"%PDF fake"

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Questão 1: Qual o valor de 2+2?"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdfplumber.open.return_value = mock_pdf

    text = download_and_extract_text("valid/file.pdf")
    assert "Questão 1" in text
