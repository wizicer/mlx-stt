from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata, get_package_paths

_package_root, package_path = get_package_paths("mlx_audio")
root = Path(package_path)
datas = copy_metadata("mlx-audio")

# mlx-audio's registry scans directories and parses kind/utils.py at runtime.
for file in root.glob("*/utils.py"):
    datas.append((str(file), "mlx_audio/" + str(file.parent.relative_to(root))))
for file in root.glob("*/models/*/__init__.py"):
    datas.append((str(file), "mlx_audio/" + str(file.parent.relative_to(root))))

hiddenimports = [
    "mlx_audio.stt.models.qwen3_asr",
    "mlx_audio.stt.models.qwen3_asr.qwen3_asr",
    "mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner",
]
