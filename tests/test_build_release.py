from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


def test_validate_mlx_binary_reference_accepts_macos15_target():
    files = {
        "libmlx.dylib": {
            "size": 21875120,
            "sha256": "0c9db159aedd2951bb77610c8b028c4ef79bc64ccc4c73b07a4452a60612ac03",
        },
        "libjaccl.dylib": {
            "size": 1555616,
            "sha256": "edabb4ca2e1292b2d02f076f999ef3fcd5420d8552233b0f3cb2a8ba0a0f843f",
        },
        "mlx.metallib": {
            "size": 135842224,
            "sha256": "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2",
        },
    }

    build_release.validate_mlx_binary_reference(files, "0.32.2", "macosx_15_0_arm64")


def test_validate_mlx_binary_reference_rejects_macos15_metallib_for_macos26():
    files = {
        "libmlx.dylib": {
            "size": 21942544,
            "sha256": "d24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8",
        },
        "libjaccl.dylib": {
            "size": 1555552,
            "sha256": "9cfd72679ff35c593a1d46fd30d995cc4a131eed15733617efb1118001e74084",
        },
        "mlx.metallib": {
            "size": 135842224,
            "sha256": "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2",
        },
    }

    with pytest.raises(RuntimeError, match="Unexpected mlx.metallib"):
        build_release.validate_mlx_binary_reference(files, "0.32.2", "macosx_26_0_arm64")


def test_validate_layout_rejects_wrong_packaged_metallib(tmp_path):
    staged = tmp_path / "mlx-stt"
    lib_dir = staged / "_internal" / "mlx" / "lib"
    lib_dir.mkdir(parents=True)
    executable = staged / "mlx-stt"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    (lib_dir / "libmlx.dylib").write_bytes(b"libmlx")
    (lib_dir / "mlx.metallib").write_bytes(b"wrong target")
    mlx_packaging = {
        "platform_tag": "macosx_26_0_arm64",
        "files": {
            "mlx.metallib": {
                "size": 182351120,
                "sha256": "dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed",
            }
        },
    }

    with pytest.raises(RuntimeError, match="Packaged mlx.metallib does not match"):
        build_release.validate_layout(staged, mlx_packaging)
