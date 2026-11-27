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
SUPPORTED_MACOS_TARGETS = ("15", "26")
MLX_BINARY_NAMES = ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib")


def macos_wheel_platform_tag(macos_target: str) -> str:
    return f"macosx_{macos_target}_0_arm64"


# macOS Metal libraries are not forward/backward interchangeable: the
# macosx_15_0 wheel lacks newer M-series kernels on macOS 26, while the
# macosx_26_0 metallib cannot be loaded by macOS 15. Keep these
# RECORD-derived checksums as release-time guardrails for the pinned
# dependency; update them deliberately when mlx-metal is upgraded.
MLX_METAL_BINARY_REFERENCE = {
    ("0.32.2", "macosx_15_0_arm64"): {
        "libmlx.dylib": (21875120, "0c9db159aedd2951bb77610c8b028c4ef79bc64ccc4c73b07a4452a60612ac03"),
        "libjaccl.dylib": (1555616, "edabb4ca2e1292b2d02f076f999ef3fcd5420d8552233b0f3cb2a8ba0a0f843f"),
        "mlx.metallib": (135842224, "2d2e729831a995a39778164c93ea1b2e8d87f7d8b1e717386d3d816f46f416b2"),
    },
    ("0.32.2", "macosx_26_0_arm64"): {
        "libmlx.dylib": (21942544, "d24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8"),
        "libjaccl.dylib": (1555552, "9cfd72679ff35c593a1d46fd30d995cc4a131eed15733617efb1118001e74084"),
        "mlx.metallib": (182351120, "dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed"),
    },
}

DEPENDENCIES = [
    "mlx",
    "mlx-metal",
    "mlx-audio",
    "transformers",
    "huggingface-hub",
    "numpy",
    "python-multipart",
    "starlette",
    "uvicorn",
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


def host_macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".", 1)[0])
    except (IndexError, ValueError):
        return 0


def require_matching_macos_host(macos_target: str) -> None:
    host_major = host_macos_major()
    if host_major != int(macos_target):
        raise RuntimeError(
            f"macOS {macos_target} artifacts must be built on a macOS {macos_target} arm64 host "
            f"because pip/uv select MLX Metal wheels for the host OS; current host is macOS {host_major}"
        )


def mlx_package_path() -> Path:
    spec = importlib.util.find_spec("mlx.core")
    if spec is None or spec.origin is None:
        raise RuntimeError("mlx.core is not importable in the build environment")
    return Path(spec.origin).resolve().parent


def validate_mlx_binary_reference(files: dict[str, dict[str, object]], mlx_metal_version: str, platform_tag: str) -> None:
    expected_files = MLX_METAL_BINARY_REFERENCE.get((mlx_metal_version, platform_tag))
    if expected_files is None:
        raise RuntimeError(
            f"No reference MLX binary hashes are recorded for mlx-metal {mlx_metal_version}/{platform_tag}; "
            "verify the wheel and update MLX_METAL_BINARY_REFERENCE"
        )
    for name, (expected_size, expected_sha256) in expected_files.items():
        actual = files.get(name)
        if actual is None:
            raise RuntimeError(f"Build environment is missing {name}")
        if actual["size"] != expected_size or actual["sha256"] != expected_sha256:
            raise RuntimeError(
                f"Unexpected {name} for the {platform_tag} build: "
                f"got {actual['size']} bytes / sha256 {actual['sha256']}, "
                f"expected {expected_size} bytes / sha256 {expected_sha256}"
            )


def validate_targeted_mlx_installation(platform_tag: str) -> dict[str, object]:
    package_path = mlx_package_path()
    files = {}
    for name in MLX_BINARY_NAMES:
        path = package_path / "lib" / name
        if not path.is_file():
            raise RuntimeError(f"Build environment is missing {name}: {path}")
        files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    mlx_metal_version = dependency_version("mlx-metal")
    validate_mlx_binary_reference(files, mlx_metal_version, platform_tag)
    return {
        "platform_tag": platform_tag,
        "selection": "matching macOS build host",
        "mlx_metal_version": mlx_metal_version,
        "files": files,
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


def validate_layout(staged: Path, mlx_packaging: dict[str, object]) -> None:
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

    metallib = internal / "mlx" / "lib" / "mlx.metallib"
    reference = mlx_packaging["files"]["mlx.metallib"]  # type: ignore[index]
    actual_size = metallib.stat().st_size
    actual_sha256 = sha256_file(metallib)
    if actual_size != reference["size"] or actual_sha256 != reference["sha256"]:
        raise RuntimeError(
            "Packaged mlx.metallib does not match the targeted "
            f"{mlx_packaging['platform_tag']} wheel: got {actual_size} bytes / sha256 {actual_sha256}, "
            f"expected {reference['size']} bytes / sha256 {reference['sha256']}"
        )


def build_release(version: str, dist_dir: Path, *, force: bool, macos_target: str) -> dict[str, object]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("mlx-stt release builds currently require macOS arm64")
    if macos_target not in SUPPORTED_MACOS_TARGETS:
        raise RuntimeError(f"Unsupported macOS target {macos_target}; expected one of: {', '.join(SUPPORTED_MACOS_TARGETS)}")
    require_matching_macos_host(macos_target)

    platform_tag = macos_wheel_platform_tag(macos_target)
    artifact_platform = f"macos{macos_target}-arm64"
    artifact_base = f"mlx-stt-{version}-{artifact_platform}"
    mlx_packaging = validate_targeted_mlx_installation(platform_tag)
    dist_dir = dist_dir.resolve()
    staged = dist_dir / artifact_base
    archive = dist_dir / f"{artifact_base}.tar.gz"
    checksum = dist_dir / f"{archive.name}.sha256"
    build_info_path = dist_dir / f"build-info-{artifact_platform}.json"
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
        validate_layout(staged, mlx_packaging)
        (staged / "VERSION").write_text(version + "\n", encoding="utf-8")
        build_info: dict[str, object] = {
            "name": "mlx-stt",
            "version": version,
            "artifact": archive.name,
            "platform": artifact_platform,
            "python": platform.python_version(),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "dependencies": versions(),
            "mlx_packaging": mlx_packaging,
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
    parser.add_argument(
        "--macos-target",
        choices=SUPPORTED_MACOS_TARGETS,
        default="26" if host_macos_major() >= 26 else "15",
        help="macOS target for the artifact (default: current host family)",
    )
    args = parser.parse_args()
    try:
        build_release(args.version, args.dist_dir, force=args.force, macos_target=args.macos_target)
    except RuntimeError as exc:
        print(f"build_release: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
