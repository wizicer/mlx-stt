from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AudioPreparationError

TARGET_SAMPLE_RATE = 16_000
PCM16_SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_width: int
    frame_rate: int
    frames: int
    compression_type: str

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.frame_rate if self.frame_rate else 0.0

    @property
    def is_target_pcm(self) -> bool:
        return (
            self.channels == 1
            and self.sample_width == PCM16_SAMPLE_WIDTH
            and self.frame_rate == TARGET_SAMPLE_RATE
            and self.compression_type == "NONE"
        )


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    converted: bool
    ffmpeg_path: str | None


def wav_info(path: Path) -> WavInfo | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return WavInfo(
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                frame_rate=handle.getframerate(),
                frames=handle.getnframes(),
                compression_type=handle.getcomptype(),
            )
    except (EOFError, OSError, wave.Error):
        return None


def duration_seconds(path: Path) -> float:
    info = wav_info(path)
    if info is None:
        raise AudioPreparationError(f"Cannot read WAV duration from {path}")
    return info.duration_seconds


def ffmpeg_executable() -> str | None:
    return shutil.which("ffmpeg")


def prepare_audio(source: Path, work_directory: Path) -> PreparedAudio:
    source = Path(source).expanduser()
    if not source.is_file():
        raise AudioPreparationError(f"Input audio does not exist: {source}")

    info = wav_info(source)
    if info is not None and info.is_target_pcm:
        return PreparedAudio(path=source, converted=False, ffmpeg_path=None)

    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        detail = (
            "it is not a readable WAV file"
            if info is None
            else (
                f"it is {info.frame_rate} Hz, {info.channels} channel(s), "
                f"{info.sample_width * 8}-bit, compression={info.compression_type!r}"
            )
        )
        raise AudioPreparationError(
            f"Input must be 16 kHz mono PCM WAV; {source} cannot be used directly because {detail}. "
            "Install ffmpeg with `brew install ffmpeg` to let mlx-stt convert other audio formats automatically."
        )

    work_directory.mkdir(parents=True, exist_ok=True)
    converted_path = work_directory / "input-16khz-mono-pcm.wav"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(converted_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "unknown ffmpeg error"
        raise AudioPreparationError(f"ffmpeg could not convert {source}: {message}")

    converted_info = wav_info(converted_path)
    if converted_info is None or not converted_info.is_target_pcm:
        raise AudioPreparationError(f"ffmpeg produced an invalid WAV file for {source}")
    return PreparedAudio(path=converted_path, converted=True, ffmpeg_path=ffmpeg)


def split_wav(source: Path, directory: Path, target_seconds: int = 120) -> list[tuple[Path, float]]:
    """Split mono PCM WAV at quiet points near the target duration."""
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")

    with wave.open(str(source), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if channels != 1 or width != PCM16_SAMPLE_WIDTH:
            raise ValueError("Expected 16-bit mono PCM WAV")
        samples = np.frombuffer(handle.readframes(frames), dtype="<i2")

    boundaries = [0]
    duration = frames / rate
    target = target_seconds
    while target < duration - 30:
        candidates = range(
            max(int((target - 8) * rate), boundaries[-1] + 30 * rate),
            min(int((target + 8) * rate), frames - 30 * rate),
            max(1, rate // 10),
        )
        if not candidates:
            break
        window = max(1, rate // 5)
        midpoint = min(
            candidates,
            key=lambda position: float(
                np.mean(
                    samples[max(0, position - window // 2) : min(frames, position + window // 2)].astype(
                        np.float32
                    )
                    ** 2
                )
            ),
        )
        boundaries.append(midpoint)
        target = midpoint / rate + target_seconds
    boundaries.append(frames)

    directory.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        path = directory / f"chunk-{index:03d}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(PCM16_SAMPLE_WIDTH)
            handle.setframerate(rate)
            handle.writeframes(samples[start:end].tobytes())
        chunks.append((path, start / rate))
    return chunks
