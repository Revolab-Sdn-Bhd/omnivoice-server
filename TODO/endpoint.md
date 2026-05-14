# SepBox API — Complete Endpoint Reference

Base URL: `http://<host>:<port>` (default port from `PORT` env var)

Audio: 24kHz mono, 16-bit PCM. Endpoints return either full WAV files, SSE streams of base64-encoded PCM chunks, or binary WebSocket frames.

---

## Quick Reference

| # | Method | Path | Streaming | Auth | Description |
|---|--------|------|-----------|------|-------------|
| 1 | GET | `/health` | No | None | Server health check |
| 2 | GET | `/voices` | No | None | List all available voices |
| 3 | GET | `/api/speakers` | No | None | List voices (legacy key `"speakers"`) |
| 4 | POST | `/voices` | No | None | Upload a new voice WAV |
| 5 | POST | `/api/create_speaker` | No | None | Upload voice (legacy param names) |
| 6 | GET | `/api/quotes` | No | None | Curated test sentences |
| 7 | POST | `/generate` | No | None | Full TTS → WAV file |
| 8 | POST | `/generate/stream` | SSE | None | Streaming TTS → SSE events |
| 9 | POST | `/api/generate_tts` | No | None | Legacy ChatterBox TTS (form data) |
| 10 | POST | `/api/generate_tts_stream` | SSE | None | Legacy ChatterBox streaming TTS (form data) |
| 11 | WS | `/tts/websocket` | Binary | None | Cartesia-compatible streaming TTS |
| 12 | GET | `/v1/models` | No | None | OpenAI-compatible model list |
| 13 | GET | `/v1/audio/voices` | No | None | OpenAI-compatible voice list |
| 14 | POST | `/v1/audio/speech` | Optional | None | OpenAI-compatible TTS |

---

## 1. GET /health

Health check — verifies T3 engine and S3Gen WorkerPool are initialized.

**Response:** `application/json`

```json
{
  "status": "ok",
  "workers_healthy": 4,
  "workers_total": 4
}
```

| Field | Type | Notes |
|-------|------|-------|
| `status` | `string` | `"ok"` when ready, `"starting"` if still booting |
| `workers_healthy` | `int` | Number of S3Gen workers reporting healthy |
| `workers_total` | `int` | Total configured workers (from `NUM_S3GEN_WORKERS`) |

---

## 2. GET /voices

List all available voice WAV files from the `voices/` directory.

**Response:** `application/json`

```json
{
  "voices": [
    {
      "id": "anwar",
      "name": "Anwar",
      "path": "voices/anwar.wav"
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `voices[].id` | `string` | Filename stem (used as voice identifier in other endpoints) |
| `voices[].name` | `string` | Human-readable name (underscores → spaces, title-cased) |
| `voices[].path` | `string` | Relative path from project root |

---

## 3. GET /api/speakers

Legacy alias for `GET /voices`. Identical response except the key is `"speakers"` instead of `"voices"`.

**Response:** `application/json`

```json
{
  "speakers": [
    {
      "id": "anwar",
      "name": "Anwar",
      "path": "voices/anwar.wav"
    }
  ]
}
```

---

## 4. POST /voices

Upload a new WAV file and register it as a voice. On success, the server preloads T3 embeddings and computes the S3Gen reference dict for the new voice.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `audio_file` | `file` | Yes | — | WAV, max 50 MB | Voice reference audio |
| `voice_name` | `string` | Yes | — | Alphanumeric, `-`, `_` only | Name for the voice (becomes filename) |

**Response:** `201 Created` → `application/json`

```json
{
  "id": "my_voice",
  "name": "My Voice",
  "path": "/abs/path/to/voices/my_voice.wav"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `id` | `string` | Sanitized filename stem |
| `name` | `string` | Human-readable name |
| `path` | `string` | Absolute path to saved WAV |

**Error codes:**

| Status | Condition |
|--------|-----------|
| 400 | `voice_name` contains no valid characters |
| 413 | File exceeds 50 MB |
| 500 | File write failed |

---

## 5. POST /api/create_speaker

Legacy alias for `POST /voices`. Identical behavior except it accepts `speaker_name` instead of `voice_name`.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | `file` | Yes | Voice reference audio |
| `speaker_name` | `string` | Yes | Name for the voice |

**Response:** Same as `POST /voices`.

---

## 6. GET /api/quotes

Returns a curated list of test sentences for TTS evaluation. Includes tone tests, digit tests, entity tests, code-mixing (EN/MY) tests, and long-form passages.

**Response:** `application/json`

```json
{
  "quotes": [
    "How are you doing today?",
    "My account number is AA212CC8819000ZZ.",
    "..."
  ]
}
```

---

## 7. POST /generate

Run the full T3 → S3Gen pipeline and return a complete WAV file.

**Content-Type:** `application/json`

**Request body — `TTSRequest`:**

```json
{
  "text": "Hello, how are you?",
  "language": "en",
  "voice_ref_path": "voices/anwar.wav",
  "voice_ref_audio": null,
  "chunk_size": 0,
  "temperature": 0.3,
  "top_p": 1.0,
  "repetition_penalty": 2.0,
  "frequency_penalty": 0.3,
  "min_p": 0.05,
  "max_tokens": 1000
}
```

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `text` | `string` | Yes | — | min 1 char, max 10,000 | Text to synthesise (raw — normalized server-side) |
| `language` | `string` | No | `"en"` | `"en"`, `"ms"`, `"mixed"` | Language for text normalization |
| `voice_ref_path` | `string` | No* | `null` | Must be in `voices/` dir | Path to voice WAV (mutually exclusive with `voice_ref_audio`) |
| `voice_ref_audio` | `string` | No* | `null` | Base64-encoded WAV | Inline voice audio (mutually exclusive with `voice_ref_path`) |
| `chunk_size` | `int` | No | `0` | ≥ 0 | Audio chunk size in frames (0 = auto) |
| `temperature` | `float` | No | `0.3` | 0.0 – 2.0 | T3 sampling temperature |
| `top_p` | `float` | No | `1.0` | 0.0 – 1.0 | T3 nucleus sampling threshold |
| `repetition_penalty` | `float` | No | `2.0` | ≥ 0.0 | T3 repetition penalty |
| `frequency_penalty` | `float` | No | `0.3` | ≥ 0.0 | T3 frequency penalty |
| `min_p` | `float` | No | `0.05` | 0.0 – 1.0 | T3 minimum probability threshold |
| `max_tokens` | `int` | No | `1000` | ≥ 1 | Maximum speech tokens to generate |

*One of `voice_ref_path` or `voice_ref_audio` should be provided.

**Response:** `audio/wav` — complete WAV file (mono, 24kHz, 16-bit)

**Error codes:**

| Status | Condition |
|--------|-----------|
| 400 | Text exceeds 10,000 characters |
| 404 | Voice path not found |
| 500 | TTS pipeline error |

---

## 8. POST /generate/stream

Stream TTS audio as Server-Sent Events. Same request body as `POST /generate`.

**Content-Type:** `application/json` (request)

**Response:** `text/event-stream` — SSE stream

Each event is formatted as `data: <JSON>\n\n`.

**Audio chunk event:**

```json
{
  "chunk_index": 0,
  "audio": "<base64-encoded float32 PCM bytes>",
  "sample_rate": 24000,
  "dtype": "float32",
  "is_final": false
}
```

**Final event:**

```json
{
  "chunk_index": 42,
  "audio": "",
  "sample_rate": 24000,
  "dtype": "float32",
  "is_final": true
}
```

**Error event:**

```json
{
  "error": "<error message>",
  "is_final": true
}
```

| Field | Type | Notes |
|-------|------|-------|
| `chunk_index` | `int` | Sequential chunk number (0-based) |
| `audio` | `string` | Base64-encoded raw float32 PCM bytes (empty on final) |
| `sample_rate` | `int` | Always `24000` |
| `dtype` | `string` | Always `"float32"` |
| `is_final` | `bool` | `false` for data chunks, `true` for final/error |

**Response headers:**

| Header | Value |
|--------|-------|
| `Cache-Control` | `no-cache` |
| `Connection` | `keep-alive` |
| `X-Accel-Buffering` | `no` |

---

## 9. POST /api/generate_tts

Legacy ChatterBox-compatible alias for `POST /generate`. Uses form-encoded fields instead of JSON. Returns WAV bytes directly.

**Content-Type:** `application/x-www-form-urlencoded` (or `multipart/form-data`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | — | Text to synthesise |
| `speaker_name` | `string` | Yes | — | Voice name (stem, path, or stem.wav — normalized automatically) |
| `preset_name` | `string` | No | `"neutral"` | Ignored (compatibility only) |
| `temperature` | `float` | No | `0.3` | T3 sampling temperature |
| `top_p` | `float` | No | `1.0` | T3 nucleus sampling |
| `repetition_penalty` | `float` | No | `2.0` | T3 repetition penalty |
| `frequency_penalty` | `float` | No | `0.3` | T3 frequency penalty |
| `min_p` | `float` | No | `0.05` | T3 minimum probability |
| `max_tokens` | `int` | No | `1000` | Maximum speech tokens |

**`speaker_name` normalization:** Accepts bare stem (`"anwar"`), with extension (`"anwar.wav"`), or full path (`"/app/voices/anwar.wav"`). Extracts the stem in all cases. `[TEMP]` prefixed names are rejected with 400.

**Response:** `audio/wav` — same as `POST /generate`

---

## 10. POST /api/generate_tts_stream

Legacy ChatterBox-compatible alias for `POST /generate/stream`. Same form fields as `POST /api/generate_tts`.

**Content-Type:** `application/x-www-form-urlencoded`

Same parameters as `POST /api/generate_tts`.

**Response:** `text/event-stream` — same SSE format as `POST /generate/stream`

Note: SSE event format matches the new format (`chunk_index`, `audio`, `sample_rate`, `dtype`, `is_final`), not the old ChatterBox format (`text_segment`, `duration`, etc.).

---

## 11. WS /tts/websocket

Cartesia-compatible WebSocket endpoint for real-time streaming TTS. Designed for compatibility with the Cartesia Sonic API and the LiveKit Cartesia plugin.

### Protocol

**Inbound — Generate message:**

```json
{
  "model_id": "sonic-2",
  "voice": {"mode": "id", "id": "anwar"},
  "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
  "language": "en",
  "context_id": "my-paragraph",
  "transcript": "Text to speak.",
  "continue": true,
  "header_mode": "none"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_id` | `string` | No | — | Ignored (server uses its own model) |
| `voice.mode` | `string` | No | `"id"` | Voice selection mode |
| `voice.id` | `string` | No | `"anwar"` | Voice ID (stem of a file in `voices/`) |
| `output_format.encoding` | `string` | No | `"pcm_s16le"` | Output encoding (always `pcm_s16le`) |
| `output_format.sample_rate` | `int` | No | `24000` | Sample rate (always 24kHz) |
| `language` | `string` | No | `"en"` | Language for text normalization |
| `context_id` | `string` | No | `""` | Client-chosen identifier for concurrent demux |
| `transcript` | `string` | Yes | — | Text to synthesise |
| `continue` | `bool` | No | `true` | `false` = end of context (triggers "done") |
| `header_mode` | `string` | No | `"none"` | Extension: `"none"`, `"first"`, `"all"` |

**Inbound — Cancel message:**

```json
{
  "cancel": true,
  "context_id": "my-paragraph"
}
```

**Outbound — Audio chunk:**

```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded int16 PCM bytes>",
  "done": false,
  "status_code": 206,
  "context_id": "my-paragraph"
}
```

**Outbound — Done:**

```json
{
  "type": "done",
  "done": true,
  "status_code": 200,
  "context_id": "my-paragraph"
}
```

**Outbound — Error:**

```json
{
  "type": "error",
  "error": "<error message>",
  "done": true,
  "status_code": 500,
  "context_id": "my-paragraph"
}
```

### Concurrency model

- Multiple `transcript` messages sharing the same `context_id` are processed concurrently (T3 batches them, S3Gen workers run in parallel)
- Audio is sent sequentially per context — sentence N+1's audio only starts after sentence N finishes
- `continue: false` or empty `transcript` signals end of context — `"done"` is sent after all pending audio
- `"cancel": true` immediately cancels all in-flight tasks for that context and sends `"done"`

### Audio encoding

This endpoint uses **int16 PCM** (`pcm_s16le`) — all other endpoints use **float32 PCM**. The conversion is `(float32 * 32767).clip(-32768, 32767).astype(int16)`.

---

## 12. GET /v1/models

OpenAI-compatible model list.

**Response:** `application/json`

```json
{
  "object": "list",
  "data": [
    {
      "id": "sepbox-tts",
      "object": "model",
      "created": 1714800000,
      "owned_by": "sepbox"
    }
  ]
}
```

---

## 13. GET /v1/audio/voices

OpenAI-compatible voice list.

**Response:** `application/json`

```json
{
  "object": "list",
  "data": [
    {
      "id": "anwar",
      "name": "Anwar"
    }
  ]
}
```

---

## 14. POST /v1/audio/speech

OpenAI-compatible TTS endpoint. Supports non-streaming (full WAV) and streaming (SSE) modes.

**Content-Type:** `application/json`

**Request body — `OpenAISpeechRequest`:**

```json
{
  "model": "sepbox-tts",
  "input": "Hello, how are you?",
  "voice": "anwar",
  "response_format": "wav",
  "speed": 1.0,
  "temperature": 0.3,
  "top_p": 1.0,
  "repetition_penalty": 2.0,
  "frequency_penalty": 0.3,
  "min_p": 0.05,
  "max_tokens": 1000,
  "chunk_size": 0
}
```

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `model` | `string` | No | `"sepbox-tts"` | — | Model ID (ignored, always uses SepBox) |
| `input` | `string` | Yes | — | min 1, max 10,000 | Text to synthesise |
| `voice` | `string` | No | `"anwar"` | Must exist in `voices/` | Voice ID (stem) |
| `response_format` | `string` | No | `"wav"` | `"wav"`, `"mp3"`, `"opus"` | Output format (only `wav` actually supported) |
| `speed` | `float` | No | `1.0` | 0.25 – 4.0 | Speed factor (accepted but not yet implemented) |
| `temperature` | `float` | No | `0.3` | 0.0 – 2.0 | T3 sampling temperature |
| `top_p` | `float` | No | `1.0` | 0.0 – 1.0 | T3 nucleus sampling |
| `repetition_penalty` | `float` | No | `2.0` | ≥ 0.0 | T3 repetition penalty |
| `frequency_penalty` | `float` | No | `0.3` | ≥ 0.0 | T3 frequency penalty |
| `min_p` | `float` | No | `0.05` | 0.0 – 1.0 | T3 minimum probability |
| `max_tokens` | `int` | No | `1000` | ≥ 1 | Maximum speech tokens |
| `chunk_size` | `int` | No | `0` | ≥ 0 | Audio chunk size (0 = auto) |

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `stream` | `bool` | `false` | Set to `true` for SSE streaming response |

### Non-streaming response (`?stream=false` or no param)

**Response:** `audio/wav` — complete WAV file (mono, 24kHz, 16-bit)

### Streaming response (`?stream=true`)

**Response:** `text/event-stream` — same SSE format as `POST /generate/stream`

```json
{"chunk_index": 0, "audio": "<base64 float32 PCM>", "sample_rate": 24000, "dtype": "float32", "is_final": false}
```

Final event has `"is_final": true` and `"audio": ""`.

**Error codes:**

| Status | Condition |
|--------|-----------|
| 400 | Input exceeds 10,000 characters |
| 404 | Voice not found |
| 500 | TTS pipeline error |

---

## Audio Format Reference

| Property | Value |
|----------|-------|
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit signed integer |
| WAV format | PCM (uncompressed) |
| Streaming dtype | `float32` (HTTP SSE) or `int16` (WebSocket) |

### Encoding by endpoint

| Endpoint | PCM dtype | Encoding in response |
|----------|-----------|---------------------|
| `POST /generate` | float32 → int16 | Raw WAV bytes |
| `POST /generate/stream` | float32 | Base64 in SSE JSON |
| `POST /api/generate_tts` | float32 → int16 | Raw WAV bytes |
| `POST /api/generate_tts_stream` | float32 | Base64 in SSE JSON |
| `WS /tts/websocket` | float32 → int16 | Base64 in JSON text frames |
| `POST /v1/audio/speech` | float32 → int16 | Raw WAV bytes |
| `POST /v1/audio/speech?stream=true` | float32 | Base64 in SSE JSON |

---

## T3 Sampling Parameters

These parameters control the T3 text-to-speech-token generation. All TTS endpoints accept them.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `temperature` | 0.3 | 0.0 – 2.0 | Higher = more random speech patterns |
| `top_p` | 1.0 | 0.0 – 1.0 | Nucleus sampling threshold (1.0 = disabled) |
| `repetition_penalty` | 2.0 | ≥ 0.0 | Penalizes repeated tokens |
| `frequency_penalty` | 0.3 | ≥ 0.0 | Penalizes frequent tokens |
| `min_p` | 0.05 | 0.0 – 1.0 | Minimum probability to keep a token |
| `max_tokens` | 1000 | ≥ 1 | Max speech tokens to generate (longer text needs more) |
| `chunk_size` | 0 | ≥ 0 | Audio chunk size in frames (0 = server auto-selects) |
