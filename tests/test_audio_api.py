import base64
import io
import math
import wave
import sys
import os

import pytest
from fastapi.testclient import TestClient

# Ensure backend package is importable from tests dir
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from main import app  # noqa: E402


def make_sine_wav_bytes(duration_s: float = 0.5, freq: float = 440.0, rate: int = 16000) -> bytes:
    """Generate a short mono 16k WAV with a sine tone and return raw bytes."""
    samples = int(duration_s * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(samples):
            t = i / rate
            val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * t))
            wf.writeframesraw(val.to_bytes(2, byteorder="little", signed=True))
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


def test_valid_wav_transcription_returns_200(client: TestClient):
    # Create session
    r = client.post("/sessions")
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    wav_bytes = make_sine_wav_bytes()
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    payload = {"speaker": "patient", "audio_base64": b64, "mime_type": "audio/wav"}
    r2 = client.post(f"/sessions/{session_id}/audio", json=payload)
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert "transcript" in data


def test_invalid_data_returns_400(client: TestClient):
    r = client.post("/sessions")
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    payload = {"speaker": "patient", "audio_base64": "not-a-valid-base64!!!", "mime_type": "audio/webm"}
    r2 = client.post(f"/sessions/{session_id}/audio", json=payload)
    assert r2.status_code == 400
