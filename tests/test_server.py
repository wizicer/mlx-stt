from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx2
from openai import AsyncOpenAI
from starlette.testclient import TestClient

from mlx_stt.server import create_app


class FakeEngine:
    def __init__(self):
        self.asr_spec = SimpleNamespace(
            alias="qwen3-asr-1.7b-8bit",
            repo_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        )
        self.aligner = object()
        self.calls = []

    def transcribe(self, input_path, *, align=True, language=None):
        self.calls.append({"path": input_path, "align": align, "language": language})
        return {
            "text": "你好",
            "language": "zh",
            "duration_seconds": 1.0,
            "segments": [
                {
                    "text": "你好",
                    "start": 0.1,
                    "end": 0.4,
                    "tokens": [
                        {"text": "你", "start": 0.1, "end": 0.2},
                        {"text": "好", "start": 0.2, "end": 0.4},
                    ],
                }
            ],
        }


def client_and_engine(**kwargs):
    engine = FakeEngine()
    return TestClient(create_app(engine, **kwargs)), engine


def wav_upload(path):
    return {"file": (path.name, path.read_bytes(), "audio/wav")}


def test_health_and_models():
    client, _engine = client_and_engine()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "qwen3-asr-1.7b-8bit"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "qwen3-asr-1.7b-8bit"


def test_health_reports_ffmpeg_availability(monkeypatch):
    monkeypatch.setattr("mlx_stt.server.ffmpeg_executable", lambda: None)
    client, _engine = client_and_engine()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ffmpeg_available"] is False


def test_transcription_uses_default_model_when_model_field_is_omitted(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, engine = client_and_engine()

    response = client.post("/v1/audio/transcriptions", files=wav_upload(source))

    assert response.status_code == 200
    assert response.json()["text"] == "你好"
    assert response.json()["usage"] == {"type": "duration", "seconds": 1.0}
    assert engine.calls[0]["align"] is False
    assert engine.calls[0]["language"] is None


def test_word_timestamps_are_returned_on_transcriptions_endpoint(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, engine = client_and_engine()

    response = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        data={
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert engine.calls[0]["align"] is True
    assert payload["task"] == "transcribe"
    assert payload["language"] == "chinese"
    assert payload["words"] == [
        {"word": "你", "start": 0.1, "end": 0.2},
        {"word": "好", "start": 0.2, "end": 0.4},
    ]
    segment = payload["segments"][0]
    for key in ("id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"):
        assert key in segment


def test_openai_compatibility_model_alias_is_accepted(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, _engine = client_and_engine()

    response = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        data={"model": "whisper-1"},
    )

    assert response.status_code == 200


def test_timestamp_granularities_require_verbose_json(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, _engine = client_and_engine()

    response = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        data={"timestamp_granularities[]": "word"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_unknown_model_returns_openai_style_error(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, _engine = client_and_engine()

    response = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        data={"model": "not-a-local-model"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "model"
    assert "Omit the model field" in error["message"]


def test_api_key_is_enforced_for_openai_endpoints(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, _engine = client_and_engine(api_key="secret")

    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    unauthorized = client.post("/v1/audio/transcriptions", files=wav_upload(source))
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        headers={"Authorization": "Bearer secret"},
    )
    assert authorized.status_code == 200


def test_text_response_format(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    client, _engine = client_and_engine()

    response = client.post(
        "/v1/audio/transcriptions",
        files=wav_upload(source),
        data={"response_format": "text"},
    )

    assert response.status_code == 200
    assert response.text == "你好"


def test_openai_sdk_can_read_verbose_word_timestamps(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.1)
    engine = FakeEngine()

    async def run():
        transport = httpx2.ASGITransport(app=create_app(engine))
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            client = AsyncOpenAI(
                base_url="http://testserver/v1",
                api_key="local",
                http_client=http_client,
            )
            with source.open("rb") as audio_file:
                return await client.audio.transcriptions.create(
                    file=audio_file,
                    model="default",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

    result = asyncio.run(run())

    assert result.text == "你好"
    assert result.words is not None
    assert [(item.word, item.start, item.end) for item in result.words] == [("你", 0.1, 0.2), ("好", 0.2, 0.4)]
    assert result.segments is not None
    assert result.segments[0].text == "你好"
    assert engine.calls[0]["align"] is True
