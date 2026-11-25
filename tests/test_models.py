from __future__ import annotations

import pytest

from mlx_stt.errors import ModelResolutionError
from mlx_stt.models import ASR_MODELS, DEFAULT_ASR_ALIAS, find_local_model, resolve_model


def make_model(path, repo_name="config"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(f'{{"name": "{repo_name}"}}', encoding="utf-8")
    return path


def test_find_direct_model_under_custom_root(tmp_path):
    spec = ASR_MODELS[DEFAULT_ASR_ALIAS]
    model = make_model(tmp_path / "Qwen3-ASR-1.7B-8bit")

    assert find_local_model(spec, tmp_path) == model


def test_find_huggingface_cache_layout(tmp_path):
    spec = ASR_MODELS[DEFAULT_ASR_ALIAS]
    snapshot = make_model(tmp_path / "models--mlx-community--Qwen3-ASR-1.7B-8bit" / "snapshots" / "abc123")

    assert find_local_model(spec, tmp_path) == snapshot


def test_explicit_model_path_must_contain_config(tmp_path):
    spec = ASR_MODELS[DEFAULT_ASR_ALIAS]

    with pytest.raises(ModelResolutionError, match="missing config.json"):
        resolve_model(spec, explicit_path=tmp_path, download=False)


def test_resolve_downloads_into_configured_cache(tmp_path, monkeypatch):
    spec = ASR_MODELS[DEFAULT_ASR_ALIAS]
    calls = []

    def fake_download(repo_id, cache_dir):
        calls.append((repo_id, cache_dir))
        return make_model(cache_dir / "downloaded")

    monkeypatch.setattr("mlx_stt.models.download_snapshot", fake_download)

    path = resolve_model(spec, cache_root=tmp_path, download=True)

    assert path == tmp_path / "downloaded"
    assert calls == [(spec.repo_id, tmp_path)]


def test_missing_model_error_is_actionable(tmp_path):
    spec = ASR_MODELS[DEFAULT_ASR_ALIAS]

    with pytest.raises(ModelResolutionError, match="mlx-stt models install"):
        resolve_model(spec, cache_root=tmp_path, download=False)
