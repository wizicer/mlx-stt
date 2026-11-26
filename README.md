# mlx-stt

在 Apple Silicon Mac 上本地运行的中文语音转文字工具，输出文本和字级时间戳。

[![CI](https://github.com/wizicer/mlx-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/wizicer/mlx-stt/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/wizicer/mlx-stt)](https://github.com/wizicer/mlx-stt/releases)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-black)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`mlx-stt` 基于 MLX、mlx-audio、Qwen3-ASR 和 Qwen3-ForcedAligner。音频只在你的 Mac 上处理，不会上传。

## 功能

- 中文语音转文字，输出结构化 JSON
- 可生成逐字开始/结束时间，适合字幕、剪辑和检索
- 通过 Homebrew 安装，不污染系统 Python 环境
- 模型只需下载一次，升级软件不会重复下载
- 非 WAV 音频可借助 `ffmpeg` 自动转换

## 环境要求

- 搭载 Apple Silicon 的 Mac
- macOS arm64
- 默认模型首次下载约 3.75 GB，请预留至少 4 GB 磁盘空间
- 可选：安装 `ffmpeg` 以支持 MP3、M4A 等非 WAV 输入

## 安装

```bash
brew install wizicer/tap/mlx-stt
```

首次安装时 Homebrew 会自动添加 `wizicer/tap`。

确认安装：

```bash
mlx-stt --version
mlx-stt probe
```

## 快速开始

```bash
mlx-stt transcribe input.wav -o result.json
```

首次运行会自动下载默认模型；下载完成后，后续转写不需要重复下载。

如果只想得到文本和分段、不需要逐字时间戳：

```bash
mlx-stt transcribe input.wav -o result.json --no-align
```

`--no-align` 会跳过时间戳模型，下载体积更小，通常也更快。

MP3、M4A、视频音轨等其他输入可以先安装 `ffmpeg`：

```bash
brew install ffmpeg
mlx-stt transcribe input.mp3 -o result.json
```

## 输出

输出文件包含完整文本、分段和可选逐字时间戳。结构示例（节选）：

```json
{
  "text": "完整转写文本",
  "duration_seconds": 12.0,
  "segments": [
    {
      "text": "分段文本",
      "start": 0.0,
      "end": 3.2,
      "tokens": [
        {
          "text": "分",
          "start": 0.0,
          "end": 0.2
        }
      ]
    }
  ]
```

也可以直接输出到终端：

```bash
mlx-stt transcribe input.wav -o -
```

## 常用命令

```bash
# 查看模型缓存状态
mlx-stt models list

# 提前下载默认模型
mlx-stt models install default

# 使用较小的 0.6B 识别模型
mlx-stt transcribe input.wav --model qwen3-asr-0.6b-8bit -o result.json

# 查看运行进度
mlx-stt transcribe input.wav -o result.json --verbose
```

## 模型下载加速

如果首次下载模型较慢，可以在下载前启用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mlx-stt models install default
```

## 隐私

- 音频不会上传，转写和时间戳对齐都在本地完成
- 软件没有遥测或云端转写服务
- 网络访问只用于下载模型权重
- `brew uninstall` 不会删除已下载的模型缓存

<details>
<summary>高级：自定义模型缓存位置</summary>

默认情况下，模型会放在 Hugging Face 共享缓存中。也可以指定独立缓存目录：

```bash
export MLX_STT_MODEL_DIR=/path/to/model-cache
mlx-stt models install default
```

已有本地模型目录时，可以直接指定：

```bash
mlx-stt transcribe input.wav \
  --model-path /path/to/Qwen3-ASR-1.7B-8bit \
  --aligner-path /path/to/Qwen3-ForcedAligner-0.6B-8bit \
  -o result.json
```

</details>

<details>
<summary>开发与发布</summary>

开发环境：

```bash
git clone git@github.com:wizicer/mlx-stt.git
cd mlx-stt
uv sync --locked --extra dev
uv run pytest
```

构建 release artifact：

```bash
uv sync --locked --extra build
uv run python scripts/build_release.py
uv run python scripts/smoke_release.py dist/mlx-stt-0.1.0-macos-arm64.tar.gz --skip-probe
```

推送与 `pyproject.toml` 版本一致的 `v*` tag 会触发 GitHub Actions 自动构建、冒烟验证并发布 Release。

</details>

## License

[MIT](LICENSE)
