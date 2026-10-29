"""
Integration tests for FastAPI endpoints using TestClient.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Enable MOCK_LLM so tests run fast and offline without downloading weights
os.environ["MOCK_LLM"] = "true"

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mock_mode"] is True


def test_analyze_email_endpoint(client):
    payload = {
        "sender": "security@company.com",
        "subject": "CRITICAL: Breach detected",
        "content": "Isolate database immediately and notify executives."
    }
    response = client.post("/analyze-email", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "summary" in data
    assert "action_items" in data
    assert data["urgency"] in ["Low", "Medium", "High", "Critical"]
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]


def test_enqueue_email_endpoint(client):
    payload = {
        "sender": "worker@system.io",
        "subject": "Routine Task",
        "content": "Backup verified."
    }
    response = client.post("/enqueue-email", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "enqueued"


def test_summarize_doc_endpoint(client):
    payload = {
        "title": "Quarterly Update",
        "text": "The quarter finished with strong growth across all product lines. Operational overhead was reduced significantly."
    }
    response = client.post("/summarize-doc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "executive_summary" in data
    assert "key_findings" in data
    assert data["total_chunks"] >= 1


def test_summarize_doc_upload_endpoint(client):
    file_content = b"Strategic roadmap for 2027: Transitioning to distributed inference nodes."
    files = {
        "file": ("roadmap.txt", file_content, "text/plain")
    }
    data = {"title": "2027 Strategic Roadmap"}
    response = client.post("/summarize-doc/upload", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert res_json["title"] == "2027 Strategic Roadmap"
    assert "executive_summary" in res_json
