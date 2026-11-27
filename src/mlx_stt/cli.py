from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from . import __version__
from .errors import MlxSttError
from .models import (
    ALIGNER_MODELS,
    ASR_MODELS,
    DEFAULT_ASR_ALIAS,
    all_model_specs,
    configured_model_root,
    directory_size,
    find_local_model,
    format_bytes,
    install_targets,
    resolve_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-stt",
        description="Local speech-to-text for macOS Apple Silicon using MLX and Qwen3-ASR.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="transcribe an audio file to JSON")
    transcribe.add_argument("input", type=Path, help="input audio; 16 kHz mono PCM WAV is used directly")
    transcribe.add_argument("-o", "--output", type=Path, required=True, help="output JSON path, or '-' for stdout")
    transcribe.add_argument("--language", default="Chinese", help="language passed to Qwen3-ASR (default: Chinese)")
    transcribe.add_argument("--model", choices=sorted(ASR_MODELS), default=DEFAULT_ASR_ALIAS, help="ASR model alias")
    transcribe.add_argument("--model-path", type=Path, help="local ASR model directory")
    transcribe.add_argument("--aligner-path", type=Path, help="local forced-aligner model directory")
    transcribe.add_argument("--no-align", action="store_true", help="skip forced alignment and token timestamps")
    transcribe.add_argument("--chunk-seconds", type=int, default=120, help=argparse.SUPPRESS)
    transcribe.add_argument("--verbose", action="store_true", help="show mlx-audio progress while transcribing")
    transcribe.set_defaults(handler=command_transcribe)

    serve = subparsers.add_parser("serve", help="run an OpenAI-compatible local transcription API")
    serve.add_argument("--host", default="127.0.0.1", help="host interface to bind (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8080, help="port to bind (default: 8080)")
    serve.add_argument("--language", default="Chinese", help="default language passed to Qwen3-ASR")
    serve.add_argument("--model", choices=sorted(ASR_MODELS), default=DEFAULT_ASR_ALIAS, help="ASR model alias")
    serve.add_argument("--model-path", type=Path, help="local ASR model directory")
    serve.add_argument("--aligner-path", type=Path, help="local forced-aligner model directory")
    serve.add_argument("--api-key", help="require this Bearer token; defaults to MLX_STT_API_KEY when set")
    serve.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="uvicorn log level (default: info)",
    )
    serve.add_argument("--chunk-seconds", type=int, default=120, help=argparse.SUPPRESS)
    serve.set_defaults(handler=command_serve)

    models = subparsers.add_parser("models", help="inspect or download model weights")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)

    models_list = model_subparsers.add_parser("list", help="list known models and local installation status")
    models_list.add_argument("--json", action="store_true", help="write machine-readable JSON")
    models_list.set_defaults(handler=command_models_list)

    install_choices = sorted(ASR_MODELS) + sorted(ALIGNER_MODELS) + ["default", "all"]
    models_install = model_subparsers.add_parser("install", help="download model weights into the configured cache")
    models_install.add_argument("model", choices=install_choices, help="model alias, 'default', or 'all'")
    models_install.set_defaults(handler=command_models_install)

    probe = subparsers.add_parser("probe", help="check MLX and Metal availability without loading model weights")
    probe.set_defaults(handler=command_probe)
    return parser


def command_probe(_args: argparse.Namespace) -> int:
    try:
        import mlx.core as mx
    except Exception as exc:
        raise MlxSttError(f"MLX could not be imported: {exc}") from exc

    metal_available = bool(mx.metal.is_available())
    if metal_available and hasattr(mx, "device_info"):
        device_info = mx.device_info()
    elif metal_available:
        device_info = mx.metal.device_info()
    else:
        device_info = {}
    try:
        mlx_version = package_version("mlx")
    except PackageNotFoundError:
        mlx_version = "unknown"
    payload = {
        "mlx_version": mlx_version,
        "metal_available": metal_available,
        "device": device_info.get("device_name"),
        "python": platform.python_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if metal_available else 1


def command_models_list(args: argparse.Namespace) -> int:
    root = configured_model_root()
    rows = []
    for spec in all_model_specs():
        path = find_local_model(spec, root)
        rows.append(
            {
                "alias": spec.alias,
                "kind": spec.kind,
                "repo_id": spec.repo_id,
                "approximate_download": spec.approximate_download,
                "installed": path is not None,
                "path": str(path) if path is not None else None,
                "size": format_bytes(directory_size(path)) if path is not None else None,
            }
        )
    if args.json:
        print(json.dumps({"cache_root": str(root), "models": rows}, ensure_ascii=False, indent=2))
        return 0

    print(f"cache root: {root}")
    for row in rows:
        status = row["size"] if row["installed"] else f"not installed (~{row['approximate_download']})"
        print(f"{row['alias']:36} {row['kind']:15} {status}")
        if row["path"]:
            print(f"  {row['path']}")
    return 0


def command_models_install(args: argparse.Namespace) -> int:
    root = configured_model_root()
    root.mkdir(parents=True, exist_ok=True)
    specs = install_targets(args.model)
    for spec in specs:
        print(f"Ensuring {spec.repo_id} ({spec.approximate_download}) in {root}", file=sys.stderr)
        path = resolve_model(spec, cache_root=root, download=True)
        print(f"{spec.alias}: {path}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise MlxSttError("--port must be between 1 and 65535")

    from .audio import ffmpeg_executable
    from .server import serve
    from .transcribe import TranscriptionEngine

    if ffmpeg_executable() is None:
        print(
            "mlx-stt: warning: ffmpeg was not found; only 16 kHz mono PCM WAV input can be processed. "
            "Install ffmpeg with `brew install ffmpeg` to support MP3, M4A and other audio formats.",
            file=sys.stderr,
        )

    engine = TranscriptionEngine(
        model_alias=args.model,
        model_path=args.model_path,
        aligner_path=args.aligner_path,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
    )
    serve(
        engine,
        host=args.host,
        port=args.port,
        api_key=args.api_key or os.environ.get("MLX_STT_API_KEY") or None,
        log_level=args.log_level,
    )
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    from .transcribe import transcribe_audio

    transcribe_audio(
        input_path=args.input,
        output_path=args.output,
        model_alias=args.model,
        model_path=args.model_path,
        aligner_path=args.aligner_path,
        align=not args.no_align,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
        verbose=args.verbose,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except MlxSttError as exc:
        print(f"mlx-stt: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("mlx-stt: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
