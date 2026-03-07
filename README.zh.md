# Canary-Serve: NVIDIA Canary ASR HTTP API

[Русский](./README.md) | **中文** | [English](./README.en.md)

Canary-Serve 是一个简洁的 FastAPI 服务器，用于部署 NVIDIA Canary 多语言 `speech-to-text` 模型。

它允许你通过唯一的 `/inference` 接口上传 WAV 文件，并在几秒钟内获取文本或字幕结果，同时最大限度利用你的 NVIDIA GPU 计算性能。

## 特性

* **单词和片段时间戳**（仅 Flash 模型支持），通过简单的参数 `timestamps=yes|no` 开启或关闭。

* **PnC 开关** - 可以选择是否生成自动标点和大小写（PnC）。

* **长音频处理** - 自动将超过 10 秒的音频切分，进行并行处理并重新拼接。

* **多种响应格式** - 支持 text、srt、vtt、json 和 verbose_json。

* **支持 GPU 的 Docker-Compose** - 默认分配所有可用 GPU，也可通过 `deploy.resources.reservations.devices` 细粒度选择设备。

* **Zero-copy 模型下载** - 通过 [huggingface_hub.snapshot_download](./canary_api/utils/download_model.py) 仅下载一次模型并本地缓存。

* **轻量级镜像** - 基于 `nvidia/cuda:12.6.1-devel-ubuntu22.04`，镜像大小约为 4.5 GB，仅包含必要的运行时依赖。

* **OpenAI 兼容 API** - 提供标准的 `/v1/audio/transcriptions` 和 `/v1/audio/translations` 端点，便于与 OpenAI SDK 和客户端集成。

## 支持的模型

* [nvidia/canary-1b](https://huggingface.co/nvidia/canary-1b)
* [nvidia/canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash)
* [nvidia/canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash)

## 支持的语言

| ISO | 语言   | 语音识别 (ASR) | 翻译目标语言   | 时间戳 (Flash) |
|-----|------|------------|----------|-------------|
| en  | 英语   | +          | de/fr/es | +           |
| de  | 德语   | +          | en       | +           |
| fr  | 法语   | +          | en       | +           |
| es  | 西班牙语 | +          | en       | +           |

> 核心 Canary 模型官方只支持以上四种语言进行语音识别和语音翻译。

## 快速开始

### Docker 一键运行

```shell
docker run --gpus all -it --rm \
  -p 9000:9000 \
  -v $(pwd)/models:/app/models \
  -e CANARY_MODEL_NAME=nvidia/canary-1b-flash \
  evilfreelancer/canary-serve:latest
```

### 使用 Docker-Compose 启动

仓库中提供了最新的 [docker-compose.dist.yml](./docker-compose.dist.yml)，默认已配置 GPU 访问。

```shell
cp docker-compose.dist.yml docker-compose.yml
docker compose up -d
```

## 环境变量

| 变量名                | 默认值                    | 说明                         |
|--------------------|------------------------|----------------------------|
| CANARY_MODEL_NAME  | nvidia/canary-1b-flash | Hugging Face 上 Canary 模型名称 |
| CANARY_MODEL_PATH  | ./models               | 本地缓存模型目录路径                 |
| CANARY_BEAM_SIZE   | 1                      | 解码时的 Beam Search 宽度        |
| CANARY_BATCH_SIZE  | 1                      | 每个请求的 batch size           |
| CANARY_PNC         | yes                    | 是否保留标点符号和大小写               |
| CANARY_TIMESTAMPTS | no                     | 是否启用时间戳                    |
| CANARY_MODEL_PRECISION | fp32              | 模型精度：fp32, fp16, 或 bf16         |
| APP_BIND           | 0.0.0.0                | 服务器绑定的 IP 地址               |
| APP_PORT           | 9000                   | 容器内部监听端口                   |
| APP_WORKERS        | 1                      | Uvicorn 的进程数量              |

## HTTP API

### 传统端点

`POST /inference`

* Content-Type: multipart/form-data
* 表单字段:
    * `file` 必须是 WAV 文件（单声道/16kHz）
    * `language` en|de|fr|es（默认 en）
    * `pnc` yes|no（默认 yes）
    * `timestamps` yes|no（默认 no，仅 Flash 模型支持）
    * `beam_size`, `batch_size`（可选，整数）
    * `response_format` json|text|srt|vtt|verbose_json（默认 text）

**示例请求**

```shell
curl http://localhost:9000/inference \
  -F file=@sample.wav \
  -F language=de \
  -F response_format=text
```

成功返回示例 (JSON 格式):

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

### OpenAI 兼容端点

Canary-Serve 提供 OpenAI 兼容的端点，便于与现有的 OpenAI 客户端和 SDK 集成。

#### 转录

`POST /v1/audio/transcriptions`

将任何支持语言的音频转录为文本。

* Content-Type: multipart/form-data
* 表单字段:
    * `file` (必需) - 音频文件 (mp3, mp4, mpeg, mpga, m4a, wav, webm)，最大 100MB
    * `model` (必需) - 模型名称（接受 "whisper-1" 以兼容）
    * `response_format` (可选) - 输出格式：json, text, srt, vtt, verbose_json（默认：json）
    * `language` (可选) - 源语言：en, de, fr, es（默认：自动检测）
    * `temperature` (可选) - 为兼容 OpenAI 而保留，但会被忽略
    * `beam_size` (可选) - 解码的束搜索宽度（默认：1）

**示例**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=whisper-1 \
  -F language=de
```

**响应**

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

#### 翻译

`POST /v1/audio/translations`

转录并将音频翻译成英语。

* Content-Type: multipart/form-data
* 表单字段:
    * `file` (必需) - 音频文件 (mp3, mp4, mpeg, mpga, m4a, wav, webm)，最大 100MB
    * `model` (必需) - 模型名称（接受 "whisper-1" 以兼容）
    * `response_format` (可选) - 输出格式：json, text, srt, vtt, verbose_json（默认：json）
    * `temperature` (可选) - 为兼容 OpenAI 而保留，但会被忽略
    * `beam_size` (可选) - 解码的束搜索宽度（默认：1）

**示例**

```shell
curl http://localhost:9000/v1/audio/translations \
  -F file=@german_sample.mp3 \
  -F model=whisper-1
```

**响应**

```json
{
  "text": "Good day, this is the AI speaking."
}
```

#### 流式传输

`POST /v1/audio/transcriptions` 配合参数 `stream=true`

以 Server-Sent Events (SSE) 格式流式传输转录结果。

* Content-Type: multipart/form-data
* 表单字段:
    * `file` (必需) - 音频文件 (mp3, mp4, mpeg, mpga, m4a, wav, webm)，最大 100MB
    * `model` (必需) - 模型名称（接受 "whisper-1" 以兼容）
    * `stream` (必需) - 设置为 `true` 以启用流式传输
    * `language` (可选) - 源语言：en, de, fr, es（默认：自动检测）
    * `temperature` (可选) - 为兼容 OpenAI 而保留，但会被忽略
    * `beam_size` (可选) - 解码的束搜索宽度（默认：1）

**示例**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F language=en \
  -F stream=true
```

**响应** (SSE 格式)

```
data: {"type": "transcript.text.delta", "delta": "Hello "}

data: {"type": "transcript.text.delta", "delta": "world "}

data: {"type": "transcript.text.done", "text": "Hello world"}
```

长音频文件会自动切分为多个片段，并在处理完成后流式返回结果。

## 授权许可

本项目代码、Dockerfile 和文档基于 MIT 许可协议发布，允许在保留原作者版权和许可证声明的前提下进行商业或私用。

## 引用方式

如果你在学术或生产环境中使用 **Canary-Serve**，请参考以下引用格式：

```text
Pavel Rykov. (2025). Canary-Serve: NVIDIA Canary ASR HTTP API (Version 1.0.0) [Computer software]. GitHub. https://github.com/EvilFreelancer/docker-canary-serve
```

BibTeX:

```bibtex
@misc{rykov2025canaryserve,
    author = {Pavel Rykov},
    title = {Canary-Serve: NVIDIA Canary ASR HTTP API},
    howpublished = {\url{https://github.com/EvilFreelancer/docker-canary-serve}},
    year = {2025},
    version = {1.0.0},
    note = {MIT License}
}
```
