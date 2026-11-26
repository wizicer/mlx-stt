# mlx-stt

`mlx-stt` 是一个面向 macOS Apple Silicon 的本地中文语音转文字命令行工具。它使用 MLX 与 `mlx-audio` 运行 Qwen3-ASR，默认再使用 Qwen3 ForcedAligner 生成字级时间戳。音频只在本地处理；网络访问只发生在显式下载模型权重时。

## 功能

- `mlx-stt transcribe input.wav -o result.json` 输出文本、分段和字级时间戳。
- `--no-align` 跳过 forced aligner，减少首次下载体积和运行耗时。
- `--help` / `--version` 不导入 MLX，启动快。
- `probe` 只检查 MLX 与 Metal，不加载模型权重。
- 模型不打进软件包，复用 Hugging Face 缓存或 `MLX_STT_MODEL_DIR`。
- 16 kHz mono PCM WAV 直接使用；其他输入在检测到 `ffmpeg` 时自动转换。
- PyInstaller onedir release，可通过私人 Homebrew tap 安装。

第一版只支持 macOS arm64，不包含 Apple 公证。模型下载体积：默认 ASR 约 2.47 GB，默认 forced aligner 约 1.28 GB。

## Homebrew 安装（发布后）

两个仓库推送并创建 GitHub Release 后：

```bash
brew tap wizicer/tap git@github.com:wizicer/homebrew-tap.git
brew install mlx-stt
```

tap 完成后也可以直接执行 `brew install wizicer/tap/mlx-stt`。使用 SSH 时需要本机 key 已添加到 GitHub 账户。


## 使用

```bash
mlx-stt --version
mlx-stt probe
mlx-stt transcribe input.wav -o result.json
mlx-stt transcribe input.wav -o result.json --no-align
mlx-stt transcribe input.wav --language Chinese -o result.json
mlx-stt models list
mlx-stt models install qwen3-asr-1.7b-8bit
mlx-stt models install qwen3-forced-aligner-0.6b-8bit
mlx-stt models install default
```

输出 JSON 包含：

- `text`：完整转写文本；
- `segments`：分段文本、起止时间和可选 `tokens` 字级时间戳；
- `duration_seconds` / `elapsed_seconds`；
- `metadata`：模型、对齐、chunk 数和输入是否经过 ffmpeg 转换。

## 模型缓存与镜像

模型路径解析顺序：

1. `--model-path` / `--aligner-path` 显式目录；
2. `MLX_STT_MODEL_DIR`；
3. `HF_HUB_CACHE` / `HF_HOME` / 标准 Hugging Face cache。

`MLX_STT_MODEL_DIR` 是缓存根目录，既支持 Hugging Face 的 `models--org--name/snapshots/...` 布局，也支持直接包含 `Qwen3-ASR-1.7B-8bit/` 这类模型目录。模型不会复制到 Homebrew 安装目录，卸载或升级 brew 包不会删除模型。

在中国网络环境建议使用 Hugging Face mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mlx-stt models install default
```

`huggingface_hub` 会读取 `HF_ENDPOINT`。不要把 token 写入仓库或 shell 历史；如需私有模型，请使用本机安全的 Hugging Face 登录状态。

## 音频输入

首选输入是 16 kHz mono PCM WAV，它会被直接读取。其他 WAV 或非 WAV 输入会调用 PATH 中的 `ffmpeg` 转成临时 16 kHz mono PCM WAV：

```bash
brew install ffmpeg
```

找不到 `ffmpeg` 时，CLI 会给出明确错误，不会静默失败或上传音频。

## 本地开发

仓库推送后建议使用 SSH clone：

```bash
git clone git@github.com:wizicer/mlx-stt.git
cd mlx-stt
```

安装依赖并运行测试：

```bash
uv python install 3.12
uv sync --extra dev
uv run mlx-stt --help
uv run mlx-stt --version
uv run pytest
```

单元测试使用 fake ASR/aligner，不下载数 GB 模型，也不需要 GPU。真实模型 smoke test：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run mlx-stt probe
uv run mlx-stt models install default
uv run mlx-stt transcribe /path/to/input.wav -o /tmp/mlx-stt-result.json
```

## 构建 release artifact

安装构建依赖：

```bash
uv sync --extra dev --extra build
```

一键构建 PyInstaller onedir tar.gz：

```bash
uv run python scripts/build_release.py
```

输出类似：

```text
dist/mlx-stt-0.1.0-macos-arm64.tar.gz
dist/mlx-stt-0.1.0-macos-arm64.tar.gz.sha256
dist/build-info.json
```

在项目外路径解包并做 smoke test：

```bash
uv run python scripts/smoke_release.py dist/mlx-stt-0.1.0-macos-arm64.tar.gz
```

可以加 `--audio /path/to/input.wav` 做真实转写；默认会先执行 `--version`、`--help` 和 `probe`。onedir 目录中的 `mlx-stt` 与 `_internal/` 必须保持相邻，不能只 symlink 或单独复制主程序。

## 0.1.0 本机验证记录

以下结果来自 Apple M4 / 16 GB 的 macOS arm64 机器，用于确认 artifact 和安装路径可运行；不是跨机器性能承诺。

- `uv run pytest`：14 个测试通过。
- artifact：`mlx-stt-0.1.0-macos-arm64.tar.gz` 约 97 MB，解包后约 268 MB；sha256 为 `f64fd94c8a25f9bb3703b4ed99d3e8bc89ac5bd999fa364a5043d25f88d18d4c`。
- `scripts/smoke_release.py` 的 `--version`、`--help` 和 `probe` 检查通过。
- 使用 276.874 秒中文 WAV 加 `--audio` 完成 frozen 转写 smoke：791 字符、3 个分段。该 M4/16 GB 环境耗时约 90 秒；早期报告中 13–15 秒来自 Apple M5 / 24 GB，不能直接比较。少量标点与字级时间戳差异也可由不同 GPU/硬件数值行为解释，代码路径与原 benchmark 的 quiet-point chunking 实现一致。
- sibling `homebrew-tap` 的本地临时 tap 验证通过：`brew install`、`brew test` 和 `mlx-stt --version` 均成功；卸载不会删除 Hugging Face 模型缓存。

## 发布流程

`.github/workflows/release.yml` 已配置自动 Release。推送与 `pyproject.toml` 版本一致的 tag 后，GitHub Actions 会在 macOS arm64 runner 上构建、smoke test、校验 sha256，并自动创建/更新 GitHub Release，上传 tar.gz、sha256 和 `build-info.json`：

```bash
git tag v0.1.0
git push origin v0.1.0
```

tag 版本必须等于 `pyproject.toml` 中的 `project.version`，否则 workflow 会失败。当前自动 Release 只更新 `mlx-stt` 仓库；跨仓库自动改写 tap 需要额外 PAT/GitHub App secret，因此当前仍在 `homebrew-tap` 仓库运行 `scripts/update_formula.py` 写入新版本、URL 和 sha256，然后提交 tap 并执行 `brew install wizicer/tap/mlx-stt` 与 `brew test mlx-stt` 验证。

## 隐私与边界

- 程序不包含上传音频、远程音频 URL 或遥测逻辑。
- 只有模型下载会访问 Hugging Face / `HF_ENDPOINT` 镜像。
- 第一版不承诺 Intel Mac、Linux、Windows 或 Apple 公证。
- 私人 tap 是二进制分发机制；若未来要进入 homebrew-core，需要重新评估源码构建和审核要求。
