# mlx-stt

在 Apple Silicon Mac 上本地运行的中文语音转文字工具，输出文本和字级时间戳。

[![CI](https://github.com/wizicer/mlx-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/wizicer/mlx-stt/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/wizicer/mlx-stt)](https://github.com/wizicer/mlx-stt/releases)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-black)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`mlx-stt` 基于 MLX、mlx-audio、Qwen3-ASR 和 Qwen3-ForcedAligner。音频只在你的 Mac 上处理，不会上传。

## 为什么是 mlx-stt

这个项目源于一次实际选型。最初我希望直接使用一个现成库，因为语音转文字看起来已经是非常成熟的能力；但在寻找和实测过程中发现，现有方案很难同时满足速度、准确率、体积和逐字时间戳的需求。

我一开始更倾向于使用轻量小巧的 Rust 实现，但现有 Rust 方案在完整功能下的性能受到明显制约。最终选择基于 MLX 的方案：体积适中，在本次对比测试中速度最快，识别准确率也处于第一梯队，并能稳定输出逐字时间戳。

## 性能对比

在 Apple M5 / 24 GB 上，使用同一段 4 分 37 秒中文语音测试，模型已预先下载：

| 方案 | 仅转写 | 转写＋逐字时间戳 | CER |
|---|---:|---:|---:|
| **[mlx-stt](https://github.com/wizicer/mlx-stt)** | **16.58 秒** | **20.39 秒** | **0.84%** |
| [OneAsr](https://github.com/eclipse005/OneAsr) | — | 52.48 秒 | 1.40% |
| [QwenASR](https://github.com/huanglizhuo/QwenASR) Rust CPU | 32.17 秒 | 54.85 秒 | 0.98% |
| [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) | — | 55.75 秒 | 0.84% |
| [OpenASR](https://github.com/QuintinShaw/openasr) 1.7B q8 | — | 162.32 秒* | 12.50% |
| [second-state qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs) | 138.03 秒 | — | 1.40% |
| [qwen3-asr-rs](https://github.com/alan890104/qwen3-asr-rs) Candle | 212.68 秒 | — | 0.84% |

<small>
测试音频长 276.874 秒；时间为串行测试中位数，不包含首次模型下载。
“—”表示该方案未测或不支持。OpenASR 输出为 segment 级时间，不是严格逐字时间戳。
完整测试条件见 <a href="benchmark.md">benchmark.md</a>。
</small>

## 功能

- 中文语音转文字，输出结构化 JSON
- 可生成逐字开始/结束时间，适合字幕、剪辑和检索
- 通过 Homebrew 安装，不污染系统 Python 环境
- 可提供 OpenAI 兼容的本地 HTTP API
- 模型只需下载一次，升级软件不会重复下载
- 非 WAV 音频可借助 `ffmpeg` 自动转换

## 环境要求

- 搭载 Apple Silicon 的 Mac
- macOS 15 或更高版本（arm64）
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

## 本地 API

启动 OpenAI 兼容的本地转写服务：

```bash
mlx-stt serve
```

服务默认监听 `127.0.0.1:8080`。启动时如果未检测到 `ffmpeg`，会立即在终端输出警告；此时只能直接处理 16 kHz mono PCM WAV 输入。`/health` 中的 `ffmpeg_available` 也会显示检测结果。

最简单的请求不需要传模型名，会使用服务启动时的默认模型：

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav
```

如果需要精确到 token/字级的时间线，不需要额外 endpoint，在同一个转写 endpoint 上增加 OpenAI 标准的 `timestamp_granularities[]=word` 参数即可：

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word'
```

OpenAI SDK 的方法签名仍要求提供 `model` 参数；传 `default` 即可使用服务端默认模型：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
result = client.audio.transcriptions.create(
    file=open("audio.wav", "rb"),
    model="default",
    response_format="verbose_json",
    timestamp_granularities=["word"],
)
print(result.text)
print(result.words)
```

转写请求默认串行处理，避免多个模型推理任务同时占用过多内存。可以用 `--api-key` 或 `MLX_STT_API_KEY` 启用 Bearer token 校验。当前支持 `json`、`text` 和 `verbose_json` 响应；暂不支持 streaming、SRT/VTT 或说话人分离。

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
uv run python scripts/build_release.py --macos-target 15
uv run python scripts/smoke_release.py dist/mlx-stt-0.1.1-macos15-arm64.tar.gz --skip-probe
```

在 macOS 26 构建机上应使用 `--macos-target 26`。MLX Metal wheel 与 macOS target
必须匹配，因此需要在对应版本的 macOS runner 上分别构建 artifact。

推送与 `pyproject.toml` 版本一致的 `v*` tag 会触发 GitHub Actions 自动构建、冒烟验证并发布 Release。

</details>

## License

[MIT](LICENSE)
