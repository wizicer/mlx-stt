from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .audio import duration_seconds, prepare_audio, split_wav
from .errors import MlxSttError
from .models import (
    ALIGNER_MODELS,
    ASR_MODELS,
    DEFAULT_ALIGNER_ALIAS,
    DEFAULT_ASR_ALIAS,
    resolve_model,
)


def load_stt_model(path: Path) -> Any:
    from mlx_audio.stt.utils import load_model

    return load_model(str(path))


def output_language(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    if normalized in {"chinese", "zh", "zh-cn", "zh-hans", "zh-hant"}:
        return "zh"
    return language


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def transcribe_audio(
    *,
    input_path: Path,
    output_path: Path,
    model_alias: str = DEFAULT_ASR_ALIAS,
    model_path: Path | None = None,
    aligner_path: Path | None = None,
    align: bool = True,
    language: str = "Chinese",
    chunk_seconds: int = 120,
    verbose: bool = False,
) -> dict[str, Any]:
    if chunk_seconds <= 0:
        raise MlxSttError("--chunk-seconds must be positive")

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    asr_spec = ASR_MODELS[model_alias]
    aligner_spec = ALIGNER_MODELS[DEFAULT_ALIGNER_ALIAS]

    with tempfile.TemporaryDirectory(prefix="mlx-stt-") as temporary_directory:
        temp_root = Path(temporary_directory)
        prepared = prepare_audio(input_path, temp_root)
        resolved_model = resolve_model(asr_spec, explicit_path=model_path, download=True)
        resolved_aligner = (
            resolve_model(aligner_spec, explicit_path=aligner_path, download=True) if align else None
        )

        started = time.monotonic()
        asr = load_stt_model(resolved_model)
        aligner = load_stt_model(resolved_aligner) if resolved_aligner is not None else None

        segments: list[dict[str, Any]] = []
        texts: list[str] = []
        asr_seconds = 0.0
        asr_tokens = 0
        chunks = split_wav(prepared.path, temp_root / "chunks", target_seconds=chunk_seconds)
        for chunk_path, offset in chunks:
            result = asr.generate(str(chunk_path), language=language, verbose=verbose)
            text = str(result.text)
            texts.append(text)
            asr_seconds += float(getattr(result, "total_time", 0.0) or 0.0)
            asr_tokens += int(getattr(result, "total_tokens", 0) or 0)
            if not text.strip():
                continue

            if aligner is not None:
                alignment = aligner.generate(str(chunk_path), text=text, language=language)
                tokens = [
                    {
                        "text": item.text,
                        "start": round(float(item.start_time) + offset, 3),
                        "end": round(float(item.end_time) + offset, 3),
                    }
                    for item in alignment.items
                ]
                if tokens:
                    segments.append(
                        {
                            "text": text,
                            "start": tokens[0]["start"],
                            "end": tokens[-1]["end"],
                            "tokens": tokens,
                        }
                    )
            else:
                segments.append(
                    {
                        "text": text,
                        "start": round(offset, 3),
                        "end": round(offset + duration_seconds(chunk_path), 3),
                    }
                )

        elapsed = time.monotonic() - started
        payload: dict[str, Any] = {
            "schema_version": 1,
            "text": "".join(texts),
            "language": output_language(language),
            "duration_seconds": duration_seconds(prepared.path),
            "elapsed_seconds": elapsed,
            "segments": segments,
            "metadata": {
                "backend": "mlx-audio",
                "model": asr_spec.repo_id,
                "aligner": aligner_spec.repo_id if align else None,
                "alignment": align,
                "asr_total_time": asr_seconds,
                "asr_total_tokens": asr_tokens,
                "chunk_count": len(chunks),
                "input_converted": prepared.converted,
            },
        }

        if str(output_path) == "-":
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            write_json_atomic(output_path, payload)
        return payload
