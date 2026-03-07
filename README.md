# Canary-Serve: NVIDIA Canary ASR HTTP API

**Русский** | [中文](./README.zh.md) | [English](./README.en.md)

Canary-Serve - это минималистичный FastAPI-сервер, позволяющий работать с многоязычными
`speech-to-text` моделями NVIDIA Canary.

Он позволяет отправить WAV-файл на единственный маршрут `/inference` и получить текстовый
результат или субтитры всего за несколько секунд - максимально эффективно используя
производительность вашей NVIDIA GPU.

## Возможности

* **Отметки времени для слов и сегментов** (только для моделей Flash) через простой флаг timestamps=yes|no.

* **Переключатель PnC** - возможность включать или отключать автоматическую пунктуацию и капитализацию (PnC).

* **Обработка длинных аудиофайлов** - автоматическое разбиение аудио длиннее 10 секунд на части, параллельная обработка
  и последующая сборка.

* **Поддержка нескольких форматов ответов** - text, srt, vtt, json и verbose_json.

* **Docker-Compose с поддержкой GPU** - по умолчанию резервируются все доступные GPU, при этом возможно тонкое
  управление выбором устройств через стандартный `deploy.resources.reservations.devices`.

* **Zero-copy скачивание моделей** - модели скачиваются один раз
  через [huggingface_hub.snapshot_download](./canary_api/utils/download_model.py) и кэшируются локально.

* **Компактный образ** - итоговый Docker-образ построен на `nvidia/cuda:12.6.1-devel-ubuntu22.04`, весит около 4.5 ГБ и
  содержит только необходимые зависимости для выполнения.

* **OpenAI-совместимый API** - стандартные эндпоинты `/v1/audio/transcriptions` и `/v1/audio/translations` для
  лёгкой интеграции с SDK и клиентами OpenAI.

## Поддерживаемые модели

* [nvidia/canary-1b](https://huggingface.co/nvidia/canary-1b)
* [nvidia/canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash)
* [nvidia/canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash)

## Поддерживаемые языки

| ISO | Язык    | ASR | Перевод  | Отметки времени (Flash) |
|-----|---------|-----|----------|-------------------------|
| en  | English | +   | de/fr/es | +                       |
| de  | German  | +   | en       | +                       |
| fr  | French  | +   | en       | +                       |
| es  | Spanish | +   | en       | +                       |

> Базовые модели Canary официально поддерживают только эти четыре языка
> как для распознавания речи (ASR), так и для перевода речи в текст.

## Быстрый старт

### Однострочник для запуска через Docker

```shell
docker run --gpus all -it --rm \
  -p 9000:9000 \
  -v $(pwd)/models:/app/models \
  -e CANARY_MODEL_NAME=nvidia/canary-1b-flash \
  evilfreelancer/canary-serve:latest
```

### Запуск через Docker-Compose

В репозитории доступен актуальный [docker-compose.dist.yml](./docker-compose.dist.yml), который автоматически
предоставляет доступ к GPU внутри контейнера.

```shell
cp docker-compose.dist.yml docker-compose.yml
docker compose up -d
```

## Переменные окружения

| Variable           | Default                | Purpose                                             |
|--------------------|------------------------|-----------------------------------------------------|
| CANARY_MODEL_NAME  | nvidia/canary-1b-flash | Название чекпойнта Canary на Hugging Face           |
| CANARY_MODEL_PATH  | ./models               | Путь к локальной директории для кэшированной модели |
| CANARY_BEAM_SIZE   | 1                      | Ширина луча при декодировании                       |
| CANARY_BATCH_SIZE  | 1                      | Размер батча на запрос                              |
| CANARY_PNC         | yes                    | yes для включения пунктуации и регистра             |
| CANARY_TIMESTAMPTS | no                     | yes для активации отметок времени                   |
| CANARY_MODEL_PRECISION | fp32              | Точность модели: fp32, fp16, или bf16               |
| APP_BIND           | 0.0.0.0                | IP-адрес для привязки сервера                       |
| APP_PORT           | 9000                   | Порт сервера внутри контейнера                      |
| APP_WORKERS        | 1                      | Количество процессов Uvicorn                        |

## HTTP API

### Legacy Endpoint

`POST /inference`

* Content-Type: multipart/form-data
* Поля формы:
    * `file` WAV-файл (моно/16 кГц), обязательно
    * `language` en|de|fr|es (по умолчанию en)
    * `pnc` yes|no (по умолчанию yes)
    * `timestamps` yes|no (по умолчанию no, доступно только для Flash-моделей)
    * `beam_size`, `batch_size` (целые числа, опционально)
    * `response_format` json|text|srt|vtt|verbose_json (по умолчанию text)

**Пример запроса**

```shell
curl http://localhost:9000/inference \
  -F file=@sample.wav \
  -F language=de \
  -F response_format=text
```

Пример успешного ответа в формате JSON:

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

### OpenAI-совместимые эндпоинты

Canary-Serve предоставляет OpenAI-совместимые эндпоинты для лёгкой интеграции с существующими клиентами OpenAI.

#### Транскрибация

`POST /v1/audio/transcriptions`

Транскрибировать аудио на любом поддерживаемом языке.

* Content-Type: multipart/form-data
* Поля формы:
    * `file` (обязательно) - Аудиофайл (mp3, mp4, mpeg, mpga, m4a, wav, webm), макс. 100MB
    * `model` (обязательно) - Название модели (принимает "whisper-1" для совместимости)
    * `response_format` (опционально) - Формат вывода: json, text, srt, vtt, verbose_json (по умолчанию: json)
    * `language` (опционально) - Исходный язык: en, de, fr, es (по умолчанию: автоопределение)
    * `temperature` (опционально) - Принимается для совместимости с OpenAI, но игнорируется
    * `beam_size` (опционально) - Ширина луча для декодирования (по умолчанию: 1)

**Пример**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=whisper-1 \
  -F language=de
```

**Ответ**

```json
{
  "text": "Guten Tag, hier spricht die KI."
}
```

#### Перевод

`POST /v1/audio/translations`

Транскрибировать и перевести аудио на английский язык.

* Content-Type: multipart/form-data
* Поля формы:
    * `file` (обязательно) - Аудиофайл (mp3, mp4, mpeg, mpga, m4a, wav, webm), макс. 100MB
    * `model` (обязательно) - Название модели (принимает "whisper-1" для совместимости)
    * `response_format` (опционально) - Формат вывода: json, text, srt, vtt, verbose_json (по умолчанию: json)
    * `temperature` (опционально) - Принимается для совместимости с OpenAI, но игнорируется
    * `beam_size` (опционально) - Ширина луча для декодирования (по умолчанию: 1)

**Пример**

```shell
curl http://localhost:9000/v1/audio/translations \
  -F file=@german_sample.mp3 \
  -F model=whisper-1
```

**Ответ**

```json
{
  "text": "Good day, this is the AI speaking."
}
```

#### Стриминг

`POST /v1/audio/transcriptions` с параметром `stream=True`

Потоковая передача результатов транскрибации в формате Server-Sent Events (SSE).

* Content-Type: multipart/form-data
* Поля формы:
    * `file` (обязательно) - Аудиофайл (mp3, mp4, mpeg, mpga, m4a, wav, webm), макс. 100MB
    * `model` (обязательно) - Название модели (принимает "whisper-1" для совместимости)
    * `stream` (обязательно) - Установите в `true` для включения стриминга
    * `language` (опционально) - Исходный язык: en, de, fr, es (по умолчанию: автоопределение)
    * `temperature` (опционально) - Принимается для совместимости с OpenAI, но игнорируется
    * `beam_size` (опционально) - Ширина луча для декодирования (по умолчанию: 1)

**Пример**

```shell
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F language=en \
  -F stream=true
```

**Ответ** (формат SSE)

```
data: {"type": "transcript.text.delta", "delta": "Hello "}

data: {"type": "transcript.text.delta", "delta": "world "}

data: {"type": "transcript.text.done", "text": "Hello world"}
```

Поддерживается автоматическое разбиение длинных аудиофайлов на части с последующей потоковой передачей результатов.

## Лицензия

Код, Dockerfile и документация этого репозитория распространяются под лицензией MIT - короткой и разрешительной
лицензией, допускающей как коммерческое, так и частное использование при условии сохранения оригинального авторского
права и текста лицензии.

## Цитирование

Если вы используете **Canary-Serve** в академических или продакшен проекта, пожалуйста, указывайте ссылку следующим
образом:

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
