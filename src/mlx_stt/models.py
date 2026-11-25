from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelResolutionError

MODEL_DIR_ENV = "MLX_STT_MODEL_DIR"
DEFAULT_ASR_ALIAS = "qwen3-asr-1.7b-8bit"
DEFAULT_ALIGNER_ALIAS = "qwen3-forced-aligner-0.6b-8bit"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    kind: str
    approximate_download: str


ASR_MODELS = {
    "qwen3-asr-1.7b-8bit": ModelSpec(
        alias="qwen3-asr-1.7b-8bit",
        repo_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        kind="asr",
        approximate_download="2.47 GB",
    ),
    "qwen3-asr-0.6b-8bit": ModelSpec(
        alias="qwen3-asr-0.6b-8bit",
        repo_id="mlx-community/Qwen3-ASR-0.6B-8bit",
        kind="asr",
        approximate_download="1.01 GB",
    ),
}
ALIGNER_MODELS = {
    DEFAULT_ALIGNER_ALIAS: ModelSpec(
        alias=DEFAULT_ALIGNER_ALIAS,
        repo_id="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        kind="forced-aligner",
        approximate_download="1.28 GB",
    )
}


def all_model_specs() -> tuple[ModelSpec, ...]:
    return (*ASR_MODELS.values(), *ALIGNER_MODELS.values())


def model_spec(alias: str) -> ModelSpec:
    for spec in all_model_specs():
        if spec.alias == alias:
            return spec
    raise KeyError(alias)


def default_hf_cache() -> Path:
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache).expanduser()
    if home := os.environ.get("HF_HOME"):
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def configured_model_root() -> Path:
    if override := os.environ.get(MODEL_DIR_ENV):
        return Path(override).expanduser()
    return default_hf_cache()


def is_model_directory(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _cache_layout_candidates(root: Path, repo_id: str) -> list[Path]:
    owner, name = repo_id.split("/", 1)
    snapshots = root / f"models--{owner}--{name}" / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(
        (path for path in snapshots.iterdir() if is_model_directory(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_local_model(spec: ModelSpec, root: Path | None = None) -> Path | None:
    root = Path(root).expanduser() if root is not None else configured_model_root()
    owner, name = spec.repo_id.split("/", 1)
    direct_candidates = [root / name, root / owner / name]
    for candidate in direct_candidates:
        if is_model_directory(candidate):
            return candidate
    for candidate in _cache_layout_candidates(root, spec.repo_id):
        return candidate
    return None


def download_snapshot(repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir)))


def resolve_model(
    spec: ModelSpec,
    *,
    explicit_path: Path | str | None = None,
    cache_root: Path | None = None,
    download: bool = True,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if is_model_directory(path):
            return path
        raise ModelResolutionError(
            f"Explicit {spec.kind} model path is not a valid model directory (missing config.json): {path}"
        )

    root = Path(cache_root).expanduser() if cache_root is not None else configured_model_root()
    if local := find_local_model(spec, root):
        return local
    if not download:
        raise ModelResolutionError(
            f"{spec.repo_id} is not installed under {root}. Run `mlx-stt models install {spec.alias}` first."
        )

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    try:
        path = download_snapshot(spec.repo_id, root)
    except Exception as exc:
        raise ModelResolutionError(
            f"Could not download {spec.repo_id} from {endpoint}. Check the network, HF_ENDPOINT mirror configuration, "
            "or pass a local model with --model-path/--aligner-path."
        ) from exc
    if not is_model_directory(path):
        raise ModelResolutionError(f"Downloaded {spec.repo_id}, but {path} is not a valid model directory")
    return path


def install_targets(name: str) -> list[ModelSpec]:
    if name == "default":
        return [ASR_MODELS[DEFAULT_ASR_ALIAS], ALIGNER_MODELS[DEFAULT_ALIGNER_ALIAS]]
    if name == "all":
        return list(all_model_specs())
    return [model_spec(name)]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{size} B"
