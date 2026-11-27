from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .audio import ffmpeg_executable
from .errors import MlxSttError
from .models import ASR_MODELS
from .transcribe import TranscriptionEngine

OPENAI_COMPATIBILITY_MODELS = {
    "whisper-1",
    "gpt-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
}
SUPPORTED_RESPONSE_FORMATS = {"json", "text", "verbose_json"}
UNSUPPORTED_RESPONSE_FORMATS = {"srt", "vtt", "diarized_json"}


class ApiRequestError(Exception):
    def __init__(self, message: str, *, param: str | None = None, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.status_code = status_code


def error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        status_code=status_code,
    )


def form_text(form: FormData, name: str, default: str | None = None) -> str | None:
    value = form.get(name)
    if value is None:
        return default
    if isinstance(value, UploadFile):
        raise ApiRequestError(f"Form field {name} must be text, not a file", param=name)
    return str(value)


def form_bool(form: FormData, name: str) -> bool:
    value = form_text(form, name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_timestamp_granularities(form: FormData) -> set[str]:
    raw_values: list[str] = []
    for name in ("timestamp_granularities[]", "timestamp_granularities"):
        for value in form.getlist(name):
            if isinstance(value, UploadFile):
                raise ApiRequestError(f"Form field {name} must be text, not a file", param=name)
            raw_values.append(str(value))

    granularities: set[str] = set()
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value:
            continue
        if value.startswith("["):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ApiRequestError(
                    "timestamp_granularities must contain 'word' or 'segment'",
                    param="timestamp_granularities",
                ) from exc
            values = decoded if isinstance(decoded, list) else [decoded]
        else:
            values = value.split(",")
        for item in values:
            granularity = str(item).strip()
            if granularity not in {"word", "segment"}:
                raise ApiRequestError(
                    "timestamp_granularities must contain only 'word' or 'segment'",
                    param="timestamp_granularities",
                )
            granularities.add(granularity)
    return granularities


def configured_model_ids(engine: TranscriptionEngine) -> set[str]:
    spec = engine.asr_spec
    return {
        "default",
        spec.alias,
        spec.alias.removesuffix("-8bit"),
        spec.repo_id,
    }


def validate_model(value: str | None, engine: TranscriptionEngine) -> None:
    if value is None or not value.strip():
        return
    normalized = value.strip()
    if normalized.lower() in OPENAI_COMPATIBILITY_MODELS:
        return
    if normalized.lower() in {item.lower() for item in configured_model_ids(engine)}:
        return

    known_models = []
    for spec in ASR_MODELS.values():
        known_models.extend((spec.alias, spec.alias.removesuffix("-8bit"), spec.repo_id))
    if normalized.lower() in {item.lower() for item in known_models}:
        raise ApiRequestError(
            f"Model {normalized!r} is not loaded by this server. Restart with `mlx-stt serve --model ...` "
            "or omit the model field to use the server default.",
            param="model",
        )
    raise ApiRequestError(
        f"Unsupported local model {normalized!r}. Omit the model field to use the server default.",
        param="model",
    )


def usage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "duration",
        "seconds": payload["duration_seconds"],
    }


def openai_language(language: str) -> str:
    return {
        "zh": "chinese",
        "en": "english",
    }.get(language, language)


def convert_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    converted = []
    for index, segment in enumerate(payload["segments"]):
        start = float(segment["start"])
        end = float(segment["end"])
        converted.append(
            {
                "id": index,
                "seek": int(round(start * 100)),
                "start": start,
                "end": end,
                "text": segment["text"],
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
        )
    return converted


def convert_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words = []
    for segment in payload["segments"]:
        for token in segment.get("tokens", []):
            words.append(
                {
                    "word": token["text"],
                    "start": float(token["start"]),
                    "end": float(token["end"]),
                }
            )
    return words


def verbose_payload(
    payload: dict[str, Any],
    *,
    include_segments: bool,
    include_words: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": "transcribe",
        "language": openai_language(str(payload["language"])),
        "duration": payload["duration_seconds"],
        "text": payload["text"],
        "usage": usage_payload(payload),
    }
    if include_segments:
        result["segments"] = convert_segments(payload)
    if include_words:
        result["words"] = convert_words(payload)
    return result


def transcription_response(
    payload: dict[str, Any],
    *,
    response_format: str,
    include_words: bool,
) -> Response:
    if response_format == "text":
        return PlainTextResponse(str(payload["text"]))
    if response_format == "json":
        return JSONResponse(
            {
                "text": payload["text"],
                "usage": usage_payload(payload),
            }
        )
    return JSONResponse(
        verbose_payload(
            payload,
            include_segments=True,
            include_words=include_words,
        )
    )


def is_authenticated(request: Request, api_key: str | None) -> bool:
    if api_key is None:
        return True
    return request.headers.get("authorization") == f"Bearer {api_key}"


def authentication_error() -> JSONResponse:
    return error_response(
        401,
        "Missing or invalid API key",
        error_type="authentication_error",
        code="invalid_api_key",
    )


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio").suffix[:32]
    descriptor, temporary_name = tempfile.mkstemp(prefix="mlx-stt-upload-", suffix=suffix)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            while chunk := await upload.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return temporary_path


def create_app(
    engine: TranscriptionEngine,
    *,
    api_key: str | None = None,
) -> Starlette:
    inference_lock = asyncio.Lock()

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": engine.asr_spec.alias,
                "alignment": engine.aligner is not None,
                "ffmpeg_available": ffmpeg_executable() is not None,
            }
        )

    async def models(request: Request) -> JSONResponse:
        if not is_authenticated(request, api_key):
            return authentication_error()
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": engine.asr_spec.alias,
                        "object": "model",
                        "created": 0,
                        "owned_by": "mlx-stt",
                    }
                ],
            }
        )

    async def transcriptions(request: Request) -> Response:
        if not is_authenticated(request, api_key):
            return authentication_error()

        uploaded_path: Path | None = None
        try:
            try:
                form = await request.form()
            except Exception as exc:
                raise ApiRequestError("Request body must be valid multipart/form-data") from exc

            if form_bool(form, "stream"):
                raise ApiRequestError("Streaming transcription is not supported", param="stream")

            response_format = (form_text(form, "response_format", "json") or "json").strip()
            if response_format in UNSUPPORTED_RESPONSE_FORMATS:
                raise ApiRequestError(f"response_format={response_format} is not supported yet", param="response_format")
            if response_format not in SUPPORTED_RESPONSE_FORMATS:
                raise ApiRequestError(f"Unsupported response_format: {response_format}", param="response_format")

            granularities = parse_timestamp_granularities(form)
            if granularities and response_format != "verbose_json":
                raise ApiRequestError(
                    "timestamp_granularities requires response_format=verbose_json",
                    param="response_format",
                )

            validate_model(form_text(form, "model"), engine)
            language = form_text(form, "language") or None

            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ApiRequestError("Missing required audio file field 'file'", param="file")
            uploaded_path = await save_upload(upload)

            include_words = "word" in granularities
            async with inference_lock:
                payload = await asyncio.to_thread(
                    engine.transcribe,
                    uploaded_path,
                    align=include_words,
                    language=language,
                )
            return transcription_response(
                payload,
                response_format=response_format,
                include_words=include_words,
            )
        except ApiRequestError as exc:
            return error_response(exc.status_code, exc.message, param=exc.param)
        except MlxSttError as exc:
            return error_response(400, str(exc))
        except Exception:
            traceback.print_exc()
            return error_response(500, "Transcription failed", error_type="server_error")
        finally:
            if uploaded_path is not None:
                uploaded_path.unlink(missing_ok=True)

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
        ]
    )


def serve(
    engine: TranscriptionEngine,
    *,
    host: str,
    port: int,
    api_key: str | None = None,
    log_level: str = "info",
) -> None:
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"} and api_key is None:
        print(
            "mlx-stt: warning: serving on a non-loopback address without an API key",
            file=sys.stderr,
        )
    app = create_app(engine, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
