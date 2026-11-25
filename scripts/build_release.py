#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dependency_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = [
    "mlx",
    "mlx-metal",
    "mlx-audio",
    "transformers",
    "huggingface-hub",
    "numpy",
    "PyInstaller",
    "pyinstaller-hooks-contrib",
]


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def versions() -> dict[str, str]:
    result = {}
    for name in DEPENDENCIES:
        try:
            result[name] = dependency_version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TRANSFORMERS_MODEL_FAMILIES = {
    "auto",
    "encoder_decoder",
    "gpt2",
    "qwen2",
    "qwen3",
    "qwen3_asr",
    "qwen3_forced_aligner",
    "whisper",
}


def transformers_package_path() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("transformers is not importable in the build environment")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def create_pruned_transformers(build_root: Path) -> Path:
    source = transformers_package_path()
    pruned_root = build_root / "pruned-python"
    destination = pruned_root / "transformers"
    for file in source.rglob("*"):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
            continue
        relative = file.relative_to(source)
        if (
            len(relative.parts) >= 3
            and relative.parts[0] == "models"
            and relative.parts[1] not in TRANSFORMERS_MODEL_FAMILIES
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "models/auto/tokenization_auto.py":
            text = file.read_text(encoding="utf-8")
            needle = "module_name = model_type_to_module_name(module_name)\n"
            if text.count(needle) != 1:
                raise RuntimeError("Could not patch transformers tokenizer auto-import for the pruned package")
            replacement = (
                needle
                + "            if importlib.util.find_spec(f\"transformers.models.{module_name}\") is None:\n"
                + "                continue\n"
            )
            target.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
        else:
            shutil.copy2(file, target)
    return pruned_root


def run_pyinstaller(build_root: Path) -> Path:
    hooks = ROOT / "packaging" / "hooks"
    entrypoint = ROOT / "packaging" / "pyinstaller_entry.py"
    pruned_python = create_pruned_transformers(build_root)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        "mlx-stt",
        "--onedir",
        "--distpath",
        str(build_root / "dist"),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root),
        "--additional-hooks-dir",
        str(hooks),
        "--paths",
        str(pruned_python),
        "--hidden-import",
        "transformers.models.auto.tokenization_auto",
        "--hidden-import",
        "transformers.models.qwen2.tokenization_qwen2",
        "--hidden-import",
        "transformers.models.whisper.feature_extraction_whisper",
        "--hidden-import",
        "safetensors",
        "--copy-metadata",
        "mlx",
        "--copy-metadata",
        "mlx-metal",
        "--copy-metadata",
        "mlx-audio",
        str(entrypoint),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pythonpath = str(pruned_python)
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    build_root.mkdir(parents=True, exist_ok=True)
    log_path = build_root / "pyinstaller-build.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"PyInstaller failed with exit code {process.returncode}\n{tail}")
    output = build_root / "dist" / "mlx-stt"
    if not output.is_dir():
        raise RuntimeError(f"PyInstaller did not create expected onedir output: {output}")
    return output


def validate_layout(staged: Path) -> None:
    executable = staged / "mlx-stt"
    if not executable.is_file():
        raise RuntimeError(f"Invalid PyInstaller onedir layout; missing: {executable}")
    internal = staged / "_internal"
    libmlx_candidates = [
        internal / "mlx" / "lib" / "libmlx.dylib",
        internal / "libmlx.dylib",
    ]
    if not any(libmlx.is_file() and (libmlx.parent / "mlx.metallib").is_file() for libmlx in libmlx_candidates):
        raise RuntimeError("Invalid MLX layout; libmlx.dylib and mlx.metallib must be in the same directory")
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"Executable bit is not set: {executable}")


def build_release(version: str, dist_dir: Path, *, force: bool) -> dict[str, object]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("mlx-stt release builds currently require macOS arm64")

    artifact_base = f"mlx-stt-{version}-macos-arm64"
    dist_dir = dist_dir.resolve()
    staged = dist_dir / artifact_base
    archive = dist_dir / f"{artifact_base}.tar.gz"
    checksum = dist_dir / f"{archive.name}.sha256"
    build_info_path = dist_dir / "build-info.json"
    for path in (staged, archive, checksum, build_info_path):
        if path.exists():
            if not force:
                raise RuntimeError(f"Output already exists; pass --force to rebuild: {path}")
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

    dist_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlx-stt-build-", dir=dist_dir) as temporary:
        build_root = Path(temporary) / "pyinstaller"
        built = run_pyinstaller(build_root)
        shutil.move(str(built), str(staged))

    try:
        validate_layout(staged)
        (staged / "VERSION").write_text(version + "\n", encoding="utf-8")
        build_info: dict[str, object] = {
            "name": "mlx-stt",
            "version": version,
            "artifact": archive.name,
            "platform": "macos-arm64",
            "python": platform.python_version(),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "dependencies": versions(),
        }
        (staged / "build-info.json").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staged, arcname=artifact_base)
        digest = sha256_file(archive)
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        build_info["sha256"] = digest
        build_info_path.write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise

    print(f"artifact: {archive}")
    print(f"sha256:   {digest}")
    print(f"checksum: {checksum}")
    print(f"metadata: {build_info_path}")
    return build_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the macOS arm64 PyInstaller onedir release artifact.")
    parser.add_argument("--version", default=project_version(), help="release version (default: pyproject version)")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist", help="output directory")
    parser.add_argument("--force", action="store_true", help="replace existing generated release outputs")
    args = parser.parse_args()
    try:
        build_release(args.version, args.dist_dir, force=args.force)
    except RuntimeError as exc:
        print(f"build_release: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
