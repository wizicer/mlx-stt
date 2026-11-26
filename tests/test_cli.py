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
    assert "transcribe" in capsys.readouterr().out
    assert newly_imported_mlx_modules(before) == []


def test_package_version_matches_pyproject():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == pyproject["project"]["version"]
