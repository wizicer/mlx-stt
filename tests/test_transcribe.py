from __future__ import annotations

import json
from types import SimpleNamespace

from mlx_stt.transcribe import transcribe_audio


class FakeAsr:
    def generate(self, _path, *, language, verbose=False):
        assert language == "Chinese"
        return SimpleNamespace(text="你好", total_time=0.1, total_tokens=2)


class FakeAligner:
    def generate(self, _path, *, text, language):
        assert text == "你好"
        assert language == "Chinese"
        return SimpleNamespace(
            items=[
                SimpleNamespace(text="你", start_time=0.1, end_time=0.2),
                SimpleNamespace(text="好", start_time=0.2, end_time=0.4),
            ]
        )


def make_model(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_transcribe_with_alignment_writes_json(tmp_path, write_pcm_wav, monkeypatch):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=1.0)
    output = tmp_path / "result.json"
    asr_path = make_model(tmp_path / "asr-model")
    aligner_path = make_model(tmp_path / "aligner-model")
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return FakeAsr() if path == asr_path else FakeAligner()

    monkeypatch.setattr("mlx_stt.transcribe.load_stt_model", fake_load)

    payload = transcribe_audio(
        input_path=source,
        output_path=output,
        model_path=asr_path,
        aligner_path=aligner_path,
        align=True,
    )

    assert loaded == [asr_path, aligner_path]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["text"] == "你好"
    assert payload["language"] == "zh"
    assert payload["metadata"]["alignment"] is True
    assert payload["metadata"]["chunk_count"] == 1
    assert payload["segments"][0]["tokens"] == [
        {"text": "你", "start": 0.1, "end": 0.2},
        {"text": "好", "start": 0.2, "end": 0.4},
    ]


def test_transcribe_no_align_does_not_load_aligner(tmp_path, write_pcm_wav, monkeypatch):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=1.0)
    output = tmp_path / "result.json"
    asr_path = make_model(tmp_path / "asr-model")
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return FakeAsr()

    monkeypatch.setattr("mlx_stt.transcribe.load_stt_model", fake_load)

    payload = transcribe_audio(
        input_path=source,
        output_path=output,
        model_path=asr_path,
        align=False,
    )

    assert loaded == [asr_path]
    assert payload["metadata"]["alignment"] is False
    assert payload["metadata"]["aligner"] is None
    assert "tokens" not in payload["segments"][0]
    assert payload["segments"][0]["start"] == 0
