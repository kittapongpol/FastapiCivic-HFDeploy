from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def make_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (96, 64), color=(180, 180, 210)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["labels"] == ["EG", "EK", "ES", "FD", "FB", "FC", "FE", "FK"]


def test_predict_returns_ranked_json() -> None:
    response = client.post(
        "/predict",
        files={"file": ("civic-fe.jpg", make_image(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "civic-fe.jpg"
    assert body["top_prediction"]["label"] in {"EG", "EK", "ES", "FD", "FB", "FC", "FE", "FK"}
    assert len(body["predictions"]) == 8


def test_rejects_non_image_payload() -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
