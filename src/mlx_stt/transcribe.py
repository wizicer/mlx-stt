from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
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


class TranscriptionEngine:
    def __init__(
        self,
        *,
        model_alias: str = DEFAULT_ASR_ALIAS,
        model_path: Path | None = None,
        aligner_path: Path | None = None,
        language: str = "Chinese",
        chunk_seconds: int = 120,
        verbose: bool = False,
        load_aligner: bool = True,
        preload: bool = True,
    ) -> None:
        if chunk_seconds <= 0:
            raise MlxSttError("--chunk-seconds must be positive")

        self.model_alias = model_alias
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.verbose = verbose
        self.asr_spec = ASR_MODELS[model_alias]
        self.aligner_spec = ALIGNER_MODELS[DEFAULT_ALIGNER_ALIAS]
        self.explicit_model_path = model_path
        self.explicit_aligner_path = aligner_path
        self.load_aligner = load_aligner
        self.resolved_model: Path | None = None
        self.resolved_aligner: Path | None = None
        self.asr = None
        self.aligner = None
        self._lock = threading.Lock()
        if preload:
            self.load()

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _resolve_locked(self) -> None:
        if self.resolved_model is None:
            self.resolved_model = resolve_model(self.asr_spec, explicit_path=self.explicit_model_path, download=True)
        if self.resolved_aligner is None and self.load_aligner:
            self.resolved_aligner = resolve_model(
                self.aligner_spec,
                explicit_path=self.explicit_aligner_path,
                download=True,
            )

    def _load_locked(self) -> None:
        self._resolve_locked()
        if self.asr is None:
            self.asr = load_stt_model(self.resolved_model)
        if self.aligner is None and self.resolved_aligner is not None:
            self.aligner = load_stt_model(self.resolved_aligner)

    def transcribe(
        self,
        input_path: Path,
        *,
        align: bool = True,
        language: str | None = None,
        verbose: bool | None = None,
    ) -> dict[str, Any]:
        if align and not self.load_aligner:
            raise MlxSttError("Forced alignment was requested, but the aligner model is not loaded")
        with self._lock:
            return self._transcribe_locked(
                Path(input_path).expanduser(),
                align=align,
                language=language or self.language,
                verbose=self.verbose if verbose is None else verbose,
            )

    def _transcribe_locked(
        self,
        input_path: Path,
        *,
        align: bool,
        language: str,
        verbose: bool,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mlx-stt-") as temporary_directory:
            temp_root = Path(temporary_directory)
            prepared = prepare_audio(input_path, temp_root)
            self._resolve_locked()

            started = time.monotonic()
            self._load_locked()
            segments: list[dict[str, Any]] = []
            texts: list[str] = []
            asr_seconds = 0.0
            asr_tokens = 0
            chunks = split_wav(prepared.path, temp_root / "chunks", target_seconds=self.chunk_seconds)
            for chunk_path, offset in chunks:
                result = self.asr.generate(str(chunk_path), language=language, verbose=verbose)
                text = str(result.text)
                texts.append(text)
                asr_seconds += float(getattr(result, "total_time", 0.0) or 0.0)
                asr_tokens += int(getattr(result, "total_tokens", 0) or 0)
                if not text.strip():
                    continue

                if align and self.aligner is not None:
                    alignment = self.aligner.generate(str(chunk_path), text=text, language=language)
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
            return {
                "schema_version": 1,
                "text": "".join(texts),
                "language": output_language(language),
                "duration_seconds": duration_seconds(prepared.path),
                "elapsed_seconds": elapsed,
                "segments": segments,
                "metadata": {
                    "backend": "mlx-audio",
                    "model": self.asr_spec.repo_id,
                    "aligner": self.aligner_spec.repo_id if align else None,
                    "alignment": align,
                    "asr_total_time": asr_seconds,
                    "asr_total_tokens": asr_tokens,
                    "chunk_count": len(chunks),
                    "input_converted": prepared.converted,
                },
            }


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
    engine = TranscriptionEngine(
        model_alias=model_alias,
        model_path=model_path,
        aligner_path=aligner_path,
        language=language,
        chunk_seconds=chunk_seconds,
        verbose=verbose,
        load_aligner=align,
        preload=False,
    )
    payload = engine.transcribe(input_path, align=align, language=language, verbose=verbose)
    output_path = Path(output_path).expanduser()
    if str(output_path) == "-":
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        write_json_atomic(output_path, payload)
    return payload
