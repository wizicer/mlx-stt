from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mlx_stt import __version__
from mlx_stt.cli import main


def newly_imported_mlx_modules(before):
    return sorted(
        name
        for name in set(sys.modules) - before
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_audio" or name.startswith("mlx_audio.")
    )


def test_version_does_not_import_mlx(capsys):
    before = set(sys.modules)

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert f"mlx-stt {__version__}" in capsys.readouterr().out
    assert newly_imported_mlx_modules(before) == []


def test_help_does_not_import_mlx(capsys):
    before = set(sys.modules)

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "transcribe" in help_output
    assert "serve" in help_output
    assert newly_imported_mlx_modules(before) == []


def test_package_version_matches_pyproject():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == pyproject["project"]["version"]


def test_serve_uses_environment_api_key(monkeypatch):
    import mlx_stt.server
    import mlx_stt.transcribe

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured["engine"] = kwargs

    def fake_serve(engine, **kwargs):
        captured["engine_instance"] = engine
        captured["serve"] = kwargs

    monkeypatch.setattr(mlx_stt.transcribe, "TranscriptionEngine", FakeEngine)
    monkeypatch.setattr(mlx_stt.server, "serve", fake_serve)
    monkeypatch.setenv("MLX_STT_API_KEY", "secret")

    assert main(["serve", "--port", "8123", "--model", "qwen3-asr-0.6b-8bit"]) == 0
    assert captured["engine"]["model_alias"] == "qwen3-asr-0.6b-8bit"
    assert captured["serve"]["port"] == 8123
    assert captured["serve"]["api_key"] == "secret"
