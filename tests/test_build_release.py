from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
spec = importlib.util.spec_from_file_location("build_release", SCRIPT)
build_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_release)


def test_create_pruned_transformers_keeps_only_qwen_runtime_families(tmp_path):
    pruned_root = build_release.create_pruned_transformers(tmp_path)

    assert (pruned_root / "transformers/models/qwen3_asr/configuration_qwen3_asr.py").is_file()
    assert (pruned_root / "transformers/models/qwen2/tokenization_qwen2.py").is_file()
    assert not (pruned_root / "transformers/models/bert").exists()

    files = [path for path in (pruned_root / "transformers").rglob("*") if path.is_file()]
    assert len(files) < 500

    tokenization_auto = (pruned_root / "transformers/models/auto/tokenization_auto.py").read_text(
        encoding="utf-8"
    )
    assert 'find_spec(f"transformers.models.{module_name}")' in tokenization_auto
