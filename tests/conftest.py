from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture
def write_pcm_wav():
    def write(
        path: Path,
        *,
        seconds: float = 1.0,
        rate: int = 16_000,
        channels: int = 1,
        frequency: float = 440.0,
    ) -> Path:
        frames = int(seconds * rate)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            samples = bytearray()
            for index in range(frames):
                value = int(1200 * math.sin(2 * math.pi * frequency * index / rate))
                for _ in range(channels):
                    samples.extend(struct.pack("<h", value))
            handle.writeframes(bytes(samples))
        return path

    return write
