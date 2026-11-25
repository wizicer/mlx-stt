from __future__ import annotations

import wave

import pytest

from mlx_stt.audio import duration_seconds, prepare_audio, split_wav, wav_info
from mlx_stt.errors import AudioPreparationError


def test_prepare_audio_uses_compliant_wav_directly(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.5)

    prepared = prepare_audio(source, tmp_path / "work")

    assert prepared.path == source
    assert prepared.converted is False
    assert prepared.ffmpeg_path is None
    assert duration_seconds(prepared.path) == pytest.approx(0.5)


def test_prepare_audio_requires_ffmpeg_for_non_target_wav(tmp_path, write_pcm_wav, monkeypatch):
    source = write_pcm_wav(tmp_path / "input.wav", seconds=0.5, rate=8_000)
    monkeypatch.setattr("mlx_stt.audio.shutil.which", lambda _name: None)

    with pytest.raises(AudioPreparationError, match="brew install ffmpeg"):
        prepare_audio(source, tmp_path / "work")


def test_wav_info_rejects_non_wav(tmp_path):
    path = tmp_path / "not-audio.bin"
    path.write_bytes(b"not a wave")

    assert wav_info(path) is None


def test_split_wav_covers_input_without_overlap(tmp_path, write_pcm_wav):
    source = write_pcm_wav(tmp_path / "long.wav", seconds=100, frequency=220.0)

    chunks = split_wav(source, tmp_path / "chunks", target_seconds=50)

    assert len(chunks) == 2
    assert chunks[0][1] == 0
    assert chunks[1][1] > 0
    total_frames = 0
    for chunk_path, _offset in chunks:
        with wave.open(str(chunk_path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16_000
            total_frames += handle.getnframes()
    assert total_frames == 100 * 16_000
