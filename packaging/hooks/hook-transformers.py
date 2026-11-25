from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata, get_package_paths

_package_root, package_path = get_package_paths("transformers")
root = Path(package_path)

# Transformers' lazy import tables inspect source files. Preserve the package
# tree while collecting modules themselves in the archive.
datas = []
for file in root.rglob("*"):
    if file.is_file() and "__pycache__" not in file.parts:
        datas.append((str(file), "transformers/" + str(file.parent.relative_to(root))))

build_only_distributions = {
    "altgraph",
    "macholib",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "setuptools",
    "pytest",
    "iniconfig",
    "pluggy",
}
for distribution in distributions():
    name = distribution.metadata.get("Name")
    if not name or name.lower() in build_only_distributions:
        continue
    try:
        datas += copy_metadata(name)
    except Exception:
        pass

hiddenimports = [
    "transformers.models.auto.tokenization_auto",
    "transformers.models.qwen2.tokenization_qwen2",
    "transformers.models.qwen3.configuration_qwen3",
    "transformers.models.qwen3_asr.configuration_qwen3_asr",
    "transformers.models.qwen3_asr.feature_extraction_qwen3_asr",
    "transformers.models.qwen3_asr.processing_qwen3_asr",
    "transformers.models.whisper.feature_extraction_whisper",
]
module_collection_mode = "pyz"
