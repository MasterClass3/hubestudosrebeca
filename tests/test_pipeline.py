import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


@patch("app.routes.health.create_client")
def test_health_endpoint(mock_create_client):
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    mock_create_client.return_value = mock_supabase

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("app.routes.pipeline._run_pipeline")
def test_process_endpoint_returns_processing(mock_run):
    response = client.post(
        "/api/pipeline/process",
        json={"pdf_upload_id": "test-uuid-1234"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["pdf_upload_id"] == "test-uuid-1234"


@patch("app.routes.pipeline.create_client")
def test_status_endpoint_not_found(mock_create_client):
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(data=None)
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/pipeline/status/nonexistent-id")
    assert response.status_code == 404


@patch("app.routes.pipeline.create_client")
def test_status_endpoint_found(mock_create_client):
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={
            "id": "test-uuid",
            "status": "completed",
            "error_message": None,
            "file_name": "prova.pdf",
            "type": "exam",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/pipeline/status/test-uuid")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_generate_analysis_empty_ids():
    response = client.post("/api/generate-analysis", json={"question_ids": []})
    assert response.status_code == 400


def test_extract_syllabus_empty_text():
    response = client.post(
        "/api/extract-syllabus",
        json={"text": "  ", "study_plan_id": "some-plan-id"},
    )
    assert response.status_code == 400
