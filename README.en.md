# Canary-Serve: NVIDIA Canary ASR HTTP API

[Русский](./README.md) | [中文](./README.zh.md) | **English**

Canary-Serve wraps NVIDIA’s Canary multilingual `speech-to-text` checkpoints
in a lean FastAPI server and Docker-Compose bundle.

It lets you stream a WAV file to a single `/inference` route and receive
plain text or subtitle formats in seconds - all while squeezing every
drop of compute out of your NVIDIA GPU.

## Features

* **Word & segment timestamps** (Flash models only) via the single `timestamps=yes|no` flag.

* **PnC toggle** - decide if punctuation & capitalisation (PnC) should be generated.

* **Chunked long-form inference** - audio longer than 10 s is auto-split, processed in parallel, and stitched back
  together.

* **Multiple response formats** - text, srt, vtt, json and verbose_json

* **GPU-aware Compose file** - reserves all GPUs by default but supports fine-grained device selection through the
  standard Compose `deploy.resources.reservations.devices` stanza.

* **Zero-copy model download** - pulls checkpoints once
  with [huggingface_hub.snapshot_download](./canary_api/utils/download_model.py), then re-uses the local cache.

* **Small footprint** - the live image is built from **nvidia/cuda:12.6.1-devel-ubuntu22.04**, weighs ~4.5 GB, and
  contains only runtime dependencies.

* **OpenAI-compatible API** - standard `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints for
  easy integration with OpenAI SDKs and clients.

## Supported models

* nvidia/canary-1b
* nvidia/canary-1b-flash
* nvidia/canary-180m-flash

## Supported Languages

| ISO | Language | ASR | Translation | Timestamps (Flash) |
|-----|----------|-----|-------------|--------------------|
| en  | English  | +   | de/fr/es    | +                  |
| de  | German   | +   | en          | +                  |
| fr  | French   | +   | en          | +                  |
| es  | Spanish  | +   | en          | +                  |

> The core Canary models officially support exactly these four languages for both speech
> recognition and speech-to-text translation.

## Quick Start

### One-shot Docker run

```shell
docker run --gpus all -it --rm \
  -p 9000:9000 \
  -v $(pwd)/models:/app/models \
  -e CANARY_MODEL_NAME=nvidia/canary-1b-flash \
  evilfreelancer/canary-serve:latest
```

### Docker-Compose

A maintained [docker-compose.dist.yml](./docker-compose.dist.yml) is included,
it automatically grants GPU access to the container.

```shell
cp docker-compose.dist.yml docker-compose.yml
docker compose up -d
```

## Environment Variables

| Variable           | Default                | Purpose                               |
|--------------------|------------------------|---------------------------------------|
| CANARY_MODEL_NAME  | nvidia/canary-1b-flash | Any Canary checkpoint on Hugging Face |
| CANARY_MODEL_PATH  | ./models               | Host-mounted cache directory          |
| CANARY_BEAM_SIZE   | 1                      | Beam width for decoding               |
| CANARY_BATCH_SIZE  | 1                      | Batch size per request                |
| CANARY_PNC         | yes                    | yes to keep punctuation + case        |
| CANARY_TIMESTAMPTS | no                     | yes to request timestamps             |
| CANARY_MODEL_PRECISION | fp32              | Model precision: fp32, fp16, or bf16  |
| APP_BIND           | 0.0.0.0                | Container bind interface              |
| APP_PORT           | 9000                   | Container port                        |
| APP_WORKERS        | 1                      | Uvicorn worker processes              |

## HTTP API

### Legacy Endpoint

`POST /inference`

* Content-Type: multipart/form-data
* Form fields
    * `file` (WAV, mono/16 kHz) required
    * `language` en|de|fr|es (default en)
    * `pnc` yes|no (default yes)
    * `timestamps` yes|no (default no, Flash only)
    * `beam_size`, `batch_size` (ints, optional)
    * `response_format` json|text|srt|vtt|verbose_json (default text)

**Example**

```shell
curl http://localhost:9000/inference \
  -F file=@sample.wav \
  -F language=de \
  -F response_format=text
```

Successful JSON response:

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

### OpenAI-Compatible Endpoints

Canary-Serve provides OpenAI-compatible endpoints for easy integration with existing OpenAI clients and SDKs.

#### Transcription

`POST /v1/audio/transcriptions`

Transcribe audio in any supported language.

* Content-Type: multipart/form-data
* Form fields:
    * `file` (required) - Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm), max 100MB
    * `model` (required) - Model name (accepts "whisper-1" for compatibility)
    * `response_format` (optional) - Output format: json, text, srt, vtt, verbose_json (default: json)
    * `language` (optional) - Source language: en, de, fr, es (default: auto-detect)
    * `temperature` (optional) - Accepted for OpenAI compatibility but ignored
    * `beam_size` (optional) - Beam size for decoding (default: 1)

**Example**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=whisper-1 \
  -F language=de
```

**Response**

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

#### Translation

`POST /v1/audio/translations`

Transcribe and translate audio to a target language.

* Content-Type: multipart/form-data
* Form fields:
    * `file` (required) - Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm), max 100MB
    * `model` (required) - Model name (accepts "whisper-1" for compatibility)
    * `response_format` (optional) - Output format: json, text, srt, vtt, verbose_json (default: json)
    * `temperature` (optional) - Accepted for OpenAI compatibility but ignored
    * `beam_size` (optional) - Beam size for decoding (default: 1)
    * `target_lang` (optional) - Target language for translation: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk (default: en)

**Example - Translate to English (default)**

```shell
curl http://localhost:9000/v1/audio/translations \
  -F file=@german_sample.mp3 \
  -F model=whisper-1
```

**Example - Translate to Spanish**

```shell
curl http://localhost:9000/v1/audio/translations \
  -F file=@german_sample.mp3 \
  -F model=whisper-1 \
  -F target_lang=es
```

**Response**

```json
{
  "text": "Good day, this is the AI speaking."
}
```

#### Streaming

`POST /v1/audio/transcriptions` with `stream=true`

Stream transcription results as Server-Sent Events (SSE).

* Content-Type: multipart/form-data
* Form fields:
    * `file` (required) - Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm), max 100MB
    * `model` (required) - Model name (accepts "whisper-1" for compatibility)
    * `stream` (required) - Set to `true` to enable streaming
    * `language` (optional) - Source language: en, de, fr, es (default: auto-detect)
    * `temperature` (optional) - Accepted for OpenAI compatibility but ignored
    * `beam_size` (optional) - Beam size for decoding (default: 1)

**Example**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F language=en \
  -F stream=true
```

**Response** (SSE format)

```
data: {"type": "transcript.text.delta", "delta": "Hello "}

data: {"type": "transcript.text.delta", "delta": "world "}

data: {"type": "transcript.text.done", "text": "Hello world"}
```

Long audio files are automatically split into chunks and streamed as results become available.

## License

This repository’s code, Dockerfiles, and documentation are released under the MIT License - a short,
permissive license that allows commercial and private use provided you include the original
copyright and license text.

## Citation

If you use *Canary-Serve* in academic or industrial work, please cite it as:

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
