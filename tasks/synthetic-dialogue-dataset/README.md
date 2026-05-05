# M-CHAT — Malay Conversational Hearing And Talking Dataset

Synthetic multi-turn spoken dialogue dataset for Bahasa Melayu + English
(code-mixed), inspired by [J-CHAT](https://arxiv.org/html/2407.15828v2).

---

## Motivation

J-CHAT is a 76k-hour Japanese spoken dialogue corpus built by collecting
YouTube/podcast audio, performing speaker diarization, BGM removal (Demucs),
and ASR transcription. It enables training of real-time spoken dialogue models
like Moshi.

**M-CHAT** takes a different approach: we **synthetically generate** dialogues
rather than collecting from the wild. This gives us:

| Aspect | J-CHAT (Collection) | M-CHAT (Synthetic) |
|--------|--------------------|--------------------|
| Source | YouTube, podcasts | LLM-generated + TTS |
| Scale | 76k hours | Target: 600+ dialogues (Phase 1) |
| Speaker labels | Diarization (noisy) | Exact (TTS voice per role) |
| Transcript quality | ASR (WER ~5-15%) | LLM ground truth |
| Channel separation | Diarization | Stereo (L=agent, R=human) |
| Language | Japanese | Bahasa Melayu + English code-mix |
| Domain coverage | Whatever's on YouTube | Controlled themes (banking, telco, etc.) |
| Pipeline | Collect → Filter → Diarize → ASR | Theme → Situation → Dialogue → TTS → Validate |

**Why synthetic?** For low-resource languages like Malay, there isn't enough
publicly available conversational audio to collect at J-CHAT scale. Synthetic
generation lets us produce clean, labelled, stereo-separated dialogues with
exact transcripts — ideal for training and evaluating spoken dialogue systems.

**Output format** is Moshi-compatible: stereo MP3 (192kbps, L=agent, R=human)
with word-level timestamps, directly usable by `torchaudio.load()`.

---

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐    ┌──────────┐
│ 1. Theme &   │───▶│ 2. Situation │───▶│ 3. Multi-turn   │───▶│ 4. Batch │───▶│ 5. ASR   │
│    Scenario  │    │    Prompt     │    │    Dialogue     │    │    TTS   │    │ Validate │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────┘    └──────────┘
   LLM prompt          LLM              LLM generates         OmnIvoice       WhisperX
   theme config     situation.json      dialogue.json         HTTP POST       alignment
```

### Async Pipeline Architecture

Stages 2-4 are merged into `run_pipeline_async.py` as a **two-stage producer-consumer**:

```
┌────────────┐     ┌────────────────┐     ┌────────────────┐
│  Producer   │────▶│ 4 LLM Workers  │────▶│ 16 TTS Workers │
│ (situations)│     │ (dialogues,    │     │ (audio,        │
│             │     │  reuse stage3) │     │  parallel turns│
└────────────┘     └────────────────┘     └────────────────┘
   asyncio.Queue          asyncio.Queue
```

- **Producer**: generates situations for all themes concurrently via LLM
- **LLM workers (×4)**: take situations → generate dialogues. Reuses existing `stage3_dialogues/` if available (no wasted LLM calls). Puts dialogue on TTS queue
- **TTS workers (×16)**: take dialogues → synthesize all turns in parallel via `asyncio.gather` → combine stereo → save manifest
- LLM and TTS run **fully decoupled** — while TTS generates audio for dialogue A, LLM generates dialogue B

`run_loop.sh` adds concurrent ASR validation (every 120s on GPU 1) while the pipeline runs.

---

## IMPORTANT: Text Normalization via revo-norm

**OmniVoice only supports spoken-form text.** It cannot read "RM1500.20" — it needs
"R M seribu lima ratus dua puluh sen".

**WhisperX transcribes back to written form** — it outputs "RM1500.20", not the spoken form.

The pipeline uses **`revo-norm`** (cloned at `tasks/synthetic-dialogue-dataset/revo-norm/`) to
handle normalization automatically. The LLM writes natural text, and `preprocess_for_tts()`
converts it to spoken form before sending to TTS.

### How each stage handles text

| Stage | What it does | Why |
|-------|-------------|-----|
| Stage 3 (LLM) | Writes natural text (e.g., "RM1500.20") | LLM writes as a human would |
| Stage 4 (TTS) | `preprocess_for_tts()` via revo-norm → spoken form | OmniVoice needs spoken form |
| Stage 4 (manifest) | Saves both `text_written` (raw) and `text_spoken` (normalized) | Both forms preserved for downstream use |
| Stage 5 (ASR) | Compares ASR output against `text_spoken` | TTS was fed spoken form, so WER/CER compares spoken form |

---

## Stage 1 — Theme & Scenario Config

Define high-level themes with parameters. Each theme produces many situations.

```json
{
  "theme": "customer_support",
  "domain": "banking",
  "language_mix": "ms_en",
  "participants": {
    "agent": { "role": "customer_service", "voice_id": "anwar" },
    "human": { "role": "customer", "voice_id": "Dania_0" }
  },
  "constraints": {
    "num_speakers": 2,
    "distinct_voices": true,
    "note": "2 speakers with distinct voices (same gender OK)",
    "min_turns": 6,
    "max_turns": 12,
    "emotions": ["neutral", "frustrated", "relieved", "polite"],
    "code_mix_ratio": 0.3
  }
}
```

Voice ID is **persistent throughout the dialogue** — once assigned, the same voice is used for all turns of that speaker. Consistent voice per role, per dialogue.

Themes to cover:
- `customer_support` — banking, telco, insurance, e-commerce
- `podcast` — tech discussion, lifestyle, current affairs
- `debate` — education, social issues, tech ethics
- `counselling` — mental health, career guidance, academic
- `medical` — clinic triage, pharmacy, follow-up
- `education` — tutor-student, parent-teacher

---

## Stage 2 — Situation Generation

LLM generates a concrete situation from the theme.

Input: theme config
Output per situation:

```json
{
  "situation_id": "cs_banking_001",
  "theme": "customer_support",
  "domain": "banking",
  "scenario": "Customer calls about unauthorized transaction of RM1,500 on 18 Feb",
  "context": "Customer noticed the charge while checking monthly statement",
  "characters": {
    "agent": {"name": "Encik Hafiz", "gender": "male"},
    "human": {"name": "Siti Nurhaliza", "gender": "female"}
  },
  "expected_emotion_arc": ["neutral", "frustrated", "relieved"],
  "difficulty": "medium"
}
```

LLM prompt should:
- Generate characters with names and genders for agent and human roles
- Give specific Malaysian entities (Maybank, Touch n Go, GrabPay)
- Include realistic BM/EN code-mixing
- Vary the emotional arc
- Include specifics: amounts, dates, account numbers

---

## Stage 3 — Multi-turn Dialogue Generation

LLM generates the actual conversation from the situation.

Input: situation
Output per dialogue:

```json
{
  "dialogue_id": "cs_banking_001_d01",
  "situation_id": "cs_banking_001",
  "turns": [
    {
      "turn": 1,
      "speaker": "agent",
      "text": "Selamat petang, Maybank Customer Service, nama saya Encik Hafiz. Ada apa yang saya boleh bantu hari ini?",
      "emotion": "polite",
      "tone": "professional"
    },
    {
      "turn": 2,
      "speaker": "human",
      "text": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak RM1,500 transaction pada 18 Februari?",
      "emotion": "frustrated",
      "tone": "concerned"
    },
    {
      "turn": 3,
      "speaker": "agent",
      "text": "Baik Puan Siti, saya check dulu. Nombor akaun Puan 1640-8821-4490 betul ya?",
      "emotion": "polite",
      "tone": "professional"
    }
  ],
  "metadata": {
    "total_turns": 8,
    "duration_estimate_s": 45,
    "language_breakdown": { "ms": 0.65, "en": 0.30, "mixed": 0.05 }
  }
}
```

LLM prompt constraints:
- LLM writes natural text (numbers, amounts, etc. as-is) — revo-norm handles TTS normalization
- Realistic Malaysian speech patterns (lah, kan, meh, tu)
- Code-mixing natural (not forced translation)
- Agent uses formal/polite register
- Human uses casual register
- **Character names and genders from the situation must be used exactly** — correct Malaysian honorifics (Encik for male, Puan/Cik for female)
- Include emotions and tone per turn
- Each turn should be 1-3 sentences max (TTS friendly)

---

## Stage 4 — Batch TTS Generation

Convert each turn's text to audio using OmnIvoice server.

**Via `POST /generate`** with concurrent requests across a server pool.

Pipeline:
1. Load dialogue JSON
2. For each turn, normalize text via `preprocess_for_tts()` (revo-norm)
3. Call TTS with speaker's voice_id and the spoken text
4. Save audio as `turn_{N}_{speaker}.wav`
5. Track timing and success per turn

Config:
```json
{
  "servers": [{"url": "http://localhost:8881", "max_concurrent": 12}],
  "default_temperature": 0.3,
  "target_lufs": -23.0,
  "trim_front_seconds": 0.0,
  "voice_pools": {
    "male": ["anwar", "Farid_595", "hendrick-6", "wafiy_5"],
    "female": ["Dania_0", "Pearl_Happy"]
  },
  "default_genders": {
    "agent": "male",
    "human": "female"
  },
  "output_dir": "output/synthetic-dialogue/",
  "max_concurrent": 8
}
```

Voice selection: for each dialogue, **one voice is randomly picked from the gender pool per role** (e.g., agent gets a random male voice, human gets a random female voice). This voice stays **consistent across all turns** within the same dialogue. Different dialogues may have different speaker pairs, but within a dialogue, the agent always sounds the same and the human always sounds the same.

Output structure:
```
output/synthetic-dialogue/
  stage1_themes.json
  stage2_situations/
    customer_support_banking.json
    customer_support_telco.json
    ...
  stage3_dialogues/
    cs_banking_001.json
    cs_banking_002.json
    ...
  stage4_audio/
    cs_banking_001_d01/
      manifest.json          ← metadata + file paths + timing
      turn_01_agent.wav      ← individual turns (mono, 24kHz)
      turn_02_human.wav
      turn_03_agent.wav
      ...
      combined.wav           ← stereo interleaved (see below)
    cs_banking_002_d01/
      ...
  stage5_validation/
    cs_banking_001_d01_validation.json
    ...
    summary.json             ← aggregate WER/CER across all dialogues
  final/
    dataset.jsonl            ← Moshi-compatible index
    data_stereo/
      cs_banking_001_d01.json   ← transcript + timestamps
      cs_banking_001_d01.mp3    ← stereo (left=agent, right=human), 192kbps
      cs_banking_002_d01.json
      cs_banking_002_d01.mp3
      ...
    stats.json               ← full dataset statistics
```

### Moshi-Compatible Final Format

The `final/` directory follows Moshi's dataset spec, using **MP3** for storage efficiency:

| Format | Size (1 min stereo) | Notes |
|--------|---------------------|-------|
| WAV (24kHz 16-bit stereo) | ~5.8 MB | Lossless, 3x larger |
| MP3 (192kbps) | ~1.44 MB | Lossy, torchaudio loads directly via ffmpeg |

`torchaudio.load()` reads MP3 natively (via ffmpeg backend), no extra dependency needed.

```
final/
├── dataset.jsonl
└── data_stereo/
    ├── cs_banking_001_d01.json
    ├── cs_banking_001_d01.mp3
    ├── cs_banking_002_d01.json
    ├── cs_banking_002_d01.mp3
    └── ...
```

**dataset.jsonl** — one entry per dialogue:
```json
{"path": "data_stereo/cs_banking_001_d01.mp3", "duration": 24.52}
{"path": "data_stereo/cs_banking_002_d01.mp3", "duration": 18.31}
```

**Each companion .json** — transcript with word-level timestamps (generated by WhisperX in Stage 5):
```json
{
  "dialogue_id": "cs_banking_001_d01",
  "theme": "customer_support",
  "domain": "banking",
  "turns": [
    {
      "turn": 1,
      "speaker": "agent",
      "channel": "left",
      "text_written": "Selamat petang, Maybank Customer Service, nama saya Encik Hafiz.",
      "text_spoken": "Selamat petang, Maybank Customer Service, nama saya Encik Hafiz.",
      "start_s": 0.0,
      "end_s": 4.21,
      "words": [
        {"word": "Selamat", "start": 0.0, "end": 0.42},
        {"word": "petang", "start": 0.43, "end": 0.81},
        {"word": "Maybank", "start": 0.83, "end": 1.35},
        {"word": "Customer", "start": 1.36, "end": 1.82},
        {"word": "Service", "start": 1.83, "end": 2.20},
        {"word": "nama", "start": 2.21, "end": 2.48},
        {"word": "saya", "start": 2.49, "end": 2.72},
        {"word": "Encik", "start": 2.73, "end": 3.10},
        {"word": "Hafiz", "start": 3.11, "end": 3.60}
      ]
    },
    {
      "turn": 2,
      "speaker": "human",
      "channel": "right",
      "text_written": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak RM1,500 transaction pada 18 Februari?",
      "text_spoken": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak R M seribu lima ratus transaction pada lapan belas Februari?",
      "start_s": 4.51,
      "end_s": 9.62,
      "words": [...]
    }
  ]
}
```

### combined.wav — Stereo Interleaved Format

**Left channel = agent, Right channel = human.** Consistent across all dialogues.

```
Turn 1 (agent):   [████████████]                    ← left channel
                  |              silence             ← right channel

Turn 2 (human):           silence  [████████████]   ← right channel
                  [████████████]                     ← left channel (silence)

Turn 3 (agent):                    [████████████]   ← left channel
                           silence                  ← right channel
```

- Each turn's audio is placed on the correct channel based on speaker role
- The other channel has silence for the same duration
- ~0.3s gap between turns for natural pacing
- Duration = sum of all turn durations + gaps
- Enables: isolate agent only, isolate human only, or listen to full dialogue

---

## Stage 5 — ASR Validation (WhisperX)

Validate generated audio matches intended text.

Pipeline:
1. **Purge stale stubs** — auto-deletes any `skipped_asr` validation files from previous `--skip-asr` runs, forcing real ASR validation
2. Run WhisperX on each generated WAV
3. Align timestamps at word level
4. Extract spoken text from manifest (use `text_spoken` — what was fed to TTS)
5. Compare ASR transcript against spoken text
6. Compute WER (Word Error Rate) and CER (Character Error Rate)
7. Flag failures above threshold, scrub failed audio (ready for retry)

```json
{
  "dialogue_id": "cs_banking_001_d01",
  "validation": {
    "turn_02_human": {
      "written": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak RM1,500 transaction pada 18 Februari?",
      "spoken_tts": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak R M seribu lima ratus transaction pada lapan belas Februari?",
      "asr_result": "Eh, saya nampak ada transaction yang saya tak kenal. Boleh check tak RM1,500 transaction pada 18 Februari?",
      "wer": 0.0,
      "cer": 0.0,
      "alignment_score": 0.98,
      "status": "pass"
    }
  },
  "overall_wer": 0.03,
  "failed_turns": [],
  "status": "pass"
}
```

Key: ASR comparison uses **spoken form** (`text_spoken`) because TTS was fed spoken-form
text. WhisperX transcribes what it hears, so comparing against the TTS input gives
the most accurate WER/CER measurement.

WhisperX config:
```json
{
  "model": "large-v3",
  "language": "ms",
  "device": "cuda",
  "compute_type": "float16",
  "wer_threshold": 0.50,
  "cer_threshold": 0.40,
  "alignment_threshold": 0.85
}
```

---

## Implementation Files

```
tasks/synthetic-dialogue-dataset/
  README.md                    ← this file
  01_generate_themes.py        ← Stage 1: generate theme configs
  run_pipeline_async.py        ← Async pipeline (stages 2-4 merged, two-stage producer-consumer)
  05_validate_asr.py           ← Stage 5: WhisperX validation + final dataset assembly
  run_loop.sh                  ← Loop controller with concurrent validation
  config.json                  ← Global pipeline config
  revo-norm/                   ← Text normalization submodule (spoken form)
```

---

## Dataset Stats Target

| Metric | Target |
|--------|--------|
| Dialogues per theme | 100+ |
| Turns per dialogue | 6-12 |
| Themes | 6+ |
| Total dialogues | 600+ |
| Language | BM primary, 20-40% EN code-mix |
| Audio format | 24kHz stereo MP3 (192kbps), left=agent, right=human |

---

## Dependencies

- OmnIvoice server (running, with voice clone speakers loaded)
- Dual LLM backends running in parallel for throughput:
  - `zai` — Anthropic-compatible API (glm-4.5-air)
  - `qwen3` — local vLLM server (Scicom-intl/Qwen3-30B-A3B-Instruct-2507-Malaysian) on port 8900
- WhisperX for ASR validation
- `jiwer` for WER computation
- `ffmpeg` for MP3 encoding
- `revo-norm` for spoken-form text normalization

Install Python dependencies:
```bash
uv pip install openai jiwer anthropic httpx
```

---

## Quick Start

```bash
# Stage 1: Generate theme configs (no LLM needed)
uv run python tasks/synthetic-dialogue-dataset/01_generate_themes.py

# Stages 2-4: Async pipeline (situations → dialogues → TTS, all concurrent)
uv run python tasks/synthetic-dialogue-dataset/run_pipeline_async.py --max-concurrent 16

# Stage 5: ASR validation + final dataset assembly
CUDA_VISIBLE_DEVICES=1 uv run python tasks/synthetic-dialogue-dataset/05_validate_asr.py

# Or run in a loop with concurrent validation until target hours reached:
bash tasks/synthetic-dialogue-dataset/run_loop.sh 1000
```

All stages are **resumable** — re-running skips already-processed items.
Existing dialogues in `stage3_dialogues/` are reused automatically (no wasted LLM calls).
