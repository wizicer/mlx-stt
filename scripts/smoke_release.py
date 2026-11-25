#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = destination / member.name
            if not str(target.resolve()).startswith(str(destination.resolve()) + "/"):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination, filter="data")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed ({process.returncode}): {' '.join(command)}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process


def smoke(archive: Path, *, skip_probe: bool, audio: Path | None, no_align: bool) -> None:
    artifact_base = archive.name.removesuffix(".tar.gz")
    with tempfile.TemporaryDirectory(prefix="mlx-stt-smoke-") as temporary:
        root = Path(temporary)
        safe_extract(archive, root)
        executable = root / artifact_base / "mlx-stt"
        if not executable.is_file():
            raise RuntimeError(f"Archive does not contain expected executable: {artifact_base}/mlx-stt")

        version = run([str(executable), "--version"]).stdout.strip()
        if "mlx-stt" not in version:
            raise RuntimeError(f"Unexpected --version output: {version}")
        print(f"version: {version}")

        help_output = run([str(executable), "--help"]).stdout
        if "transcribe" not in help_output:
            raise RuntimeError("--help output does not contain the transcribe command")
        print("help: ok")

        if not skip_probe:
            probe_output = run([str(executable), "probe"]).stdout
            probe = json.loads(probe_output)
            if not probe.get("metal_available"):
                raise RuntimeError(f"Metal is unavailable according to probe: {probe_output}")
            print(f"probe: mlx={probe.get('mlx_version')} device={probe.get('device')}")

        if audio is not None:
            output = root / "transcription.json"
            command = [str(executable), "transcribe", str(audio), "-o", str(output)]
            if no_align:
                command.append("--no-align")
            run(command)
            payload = json.loads(output.read_text(encoding="utf-8"))
            if not payload.get("text"):
                raise RuntimeError("Smoke transcription produced empty text")
            print(f"transcribe: {len(payload['text'])} characters, {len(payload.get('segments', []))} segment(s)")

    print("smoke: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test an mlx-stt tar.gz release outside the source tree.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--skip-probe", action="store_true", help="skip MLX/Metal probe")
    parser.add_argument("--audio", type=Path, help="optional audio file for a real transcription smoke test")
    parser.add_argument("--no-align", action="store_true", help="skip forced alignment when --audio is used")
    args = parser.parse_args()
    try:
        smoke(args.archive.expanduser().resolve(), skip_probe=args.skip_probe, audio=args.audio, no_align=args.no_align)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"smoke_release: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
