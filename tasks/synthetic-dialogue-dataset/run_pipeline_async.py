#!/usr/bin/env python3
"""
Unified async pipeline: interleave LLM dialogue generation with TTS generation.

Architecture:
- Producer: generates situations via LLM (serialized, rate-limited)
- Consumer workers: pick situations → generate dialogue (LLM) → TTS all turns
- Multiple consumers run TTS concurrently across GPUs
- While one consumer waits for TTS, another can use the LLM

Usage:
    ZAI_API_KEY=... uv run python run_pipeline_async.py
    ZAI_API_KEY=... uv run python run_pipeline_async.py --max-concurrent 8
"""

import argparse
import asyncio
import itertools
import json
import logging
import random
import re
import struct
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 24_000
INT16_MAX = 32767


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.json") as f:
        return json.load(f)


# ── Text ──────────────────────────────────────────────────────────────────────


def preprocess_for_tts(text: str, language: str = "ms") -> str:
    """Normalize text with revo-norm for TTS."""
    from revo_norm import normalize_text
    return normalize_text(text, language=language)


# ── Audio ─────────────────────────────────────────────────────────────────────


def parse_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    data = wav_bytes[44:]
    pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / INT16_MAX


def write_wav(path: Path, audio: np.ndarray, channels: int = 1) -> None:
    pcm = (audio * INT16_MAX).clip(-INT16_MAX, INT16_MAX).astype(np.int16)
    raw = pcm.tobytes()
    data_size = len(raw)
    fmt_chunk = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, channels, SAMPLE_RATE, SAMPLE_RATE * channels * 2, channels * 2, 16)  # noqa: E501
    data_chunk = struct.pack('<4sI', b'data', data_size)
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    with open(path, 'wb') as f:
        f.write(header + fmt_chunk + data_chunk)
        f.write(raw)


def write_wav_stereo(path: Path, stereo: np.ndarray) -> None:
    pcm = (stereo.T * INT16_MAX).clip(-INT16_MAX, INT16_MAX).astype(np.int16)
    raw = pcm.tobytes()
    channels = 2
    data_size = len(raw)
    fmt_chunk = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, channels, SAMPLE_RATE, SAMPLE_RATE * channels * 2, channels * 2, 16)  # noqa: E501
    data_chunk = struct.pack('<4sI', b'data', data_size)
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    with open(path, 'wb') as f:
        f.write(header + fmt_chunk + data_chunk)
        f.write(raw)


def apply_fade(audio: np.ndarray, fade_ms: int = 20) -> np.ndarray:
    """Apply fade-in and fade-out to avoid pop/click at boundaries."""
    fade_samples = int(fade_ms * SAMPLE_RATE / 1000)
    if len(audio) < fade_samples * 2:
        return audio
    faded = audio.copy()
    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    faded[:fade_samples] *= fade_in
    faded[-fade_samples:] *= fade_out
    return faded


def build_stereo(
    turn_audios: list[tuple[str, np.ndarray]], gap_samples: int,
) -> np.ndarray:
    channels = []
    silence = np.zeros(gap_samples, dtype=np.float32)
    for ch_speaker in ("agent", "human"):
        parts = []
        for speaker, audio in turn_audios:
            parts.append(
                audio if speaker == ch_speaker
                else np.zeros(len(audio), dtype=np.float32)
            )
            parts.append(silence)
        if parts:
            parts.pop()
        channels.append(np.concatenate(parts))
    max_len = max(len(c) for c in channels)
    return np.stack([np.pad(c, (0, max_len - len(c))) for c in channels])


def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> bool:
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error("ffmpeg failed: %s", e)
        return False


# ── TTS Server Pool ───────────────────────────────────────────────────────────


class ServerPool:
    def __init__(self, servers: list[dict], trim_seconds: float = 0.0) -> None:
        self.trim_seconds = trim_seconds
        self.clients: list[httpx.AsyncClient] = []
        self.semaphores: list[asyncio.Semaphore] = []
        self._counter = itertools.count()
        for srv in servers:
            mc = srv.get("max_concurrent", 4)
            self.semaphores.append(asyncio.Semaphore(mc))
            self.clients.append(httpx.AsyncClient(
                base_url=srv["url"],
                limits=httpx.Limits(
                    max_connections=mc,
                    max_keepalive_connections=0,
                ),
            ))

    async def aclose(self) -> None:
        for c in self.clients:
            await c.aclose()

    async def synthesize(self, text: str, voice_id: str, lang: str = "ms") -> bytes:
        """Try each server until one succeeds."""
        last_err = None
        for attempt in range(len(self.clients) * 2):
            idx = next(self._counter) % len(self.clients)
            try:
                async with self.semaphores[idx]:
                    resp = await self.clients[idx].post(
                        "/generate",
                        json={
                            "text": text,
                            "language": lang,
                            "voice_ref_path": voice_id,
                            "trim_front_seconds": self.trim_seconds,
                            "num_step": random.randint(16, 32),
                        },
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    return resp.content
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_err = e
                logger.debug("Server %d failed (attempt %d): %s", idx, attempt + 1, e)
                continue
        raise RuntimeError(f"All TTS servers failed: {last_err}")


# ── LLM (async, serialized to respect rate limits) ────────────────────────────


class LLMClient:
    def __init__(self, llm_config: dict) -> None:
        import os
        self.name = llm_config.get("name", "zai")
        self.model = llm_config["model"]
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self._client = Anthropic(
            base_url=llm_config.get("api_base", "https://api.z.ai/api/anthropic"),
            api_key=os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
        )
        self._sem = asyncio.Semaphore(4)

    async def generate(self, prompt: str) -> str:
        async with self._sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._call, prompt)

    def _call(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception as e:
                logger.warning("[%s] attempt %d: %s", self.name, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"[{self.name}] failed after 3 attempts")


class VLLMClient:
    """OpenAI-compatible client for local vLLM server."""

    def __init__(self, config: dict) -> None:
        self.name = config.get("name", "vllm")
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 4096)
        base_url = config.get("api_base", "http://localhost:8900/v1")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._sem = asyncio.Semaphore(8)

    async def generate(self, prompt: str) -> str:
        async with self._sem:
            for attempt in range(3):
                try:
                    resp = await self._client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "max_tokens": self.max_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    # Strip <think/> blocks from Qwen3 reasoning
                    text = re.sub(r"<think[\s\S]*?</think\s*>", "", text).strip()
                    return text
                except Exception as e:
                    logger.warning("[%s] attempt %d: %s", self.name, attempt + 1, e)
                    await asyncio.sleep(3 * (attempt + 1))
            raise RuntimeError(f"[{self.name}] failed after 3 attempts")

    async def aclose(self) -> None:
        await self._client.aclose()


class LLMRouter:
    """Distributes generation across multiple LLM backends."""

    def __init__(self, backends: list[dict]) -> None:
        self._clients: list[LLMClient | VLLMClient] = []
        self._counter = itertools.count()
        for cfg in backends:
            client_type = cfg.get("type", "anthropic")
            if client_type == "openai":
                self._clients.append(VLLMClient(cfg))
            else:
                self._clients.append(LLMClient(cfg))
        logger.info("LLM Router initialized with %d backends: %s",
                     len(self._clients), [c.name for c in self._clients])

    def get_primary(self) -> LLMClient | VLLMClient:
        """Get the primary (first) backend for situation generation."""
        return self._clients[0]

    async def generate(self, prompt: str) -> tuple[str, str]:
        """Round-robin across backends. Returns (text, backend_name)."""
        idx = next(self._counter) % len(self._clients)
        client = self._clients[idx]
        text = await client.generate(prompt)
        return text, client.name

    async def aclose(self) -> None:
        for c in self._clients:
            if hasattr(c, "aclose"):
                await c.aclose()


def parse_json_response(text: str) -> dict | list:
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    parsed = json.loads(text.strip())
    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        # Find a dialogue wrapper dict (has "turns" key)
        for item in parsed:
            if isinstance(item, dict) and "turns" in item:
                return item
        # Flat array of turn objects — wrap into dialogue dict
        if all(isinstance(t, dict) and "turn" in t and "speaker" in t for t in parsed):
            return {"turns": parsed}
    return parsed


# ── Prompts ───────────────────────────────────────────────────────────────────

SITUATION_PROMPT = """\
You are generating training data for a Malay spoken dialogue system (M-CHAT).

Random seed for this batch: {seed}

Given the theme config below, generate {n} diverse, realistic situations.

Requirements:
- Each situation must be specific with concrete details
  (amounts in RM, dates, account numbers, phone numbers, IC numbers)
- Use Malaysian entities from the config where applicable
- Code-mixing between Bahasa Melayu and English naturally
- Vary difficulty (easy/medium/hard) and emotional arcs
- Exactly 2 speakers: agent and human, with distinct voices (prefer different genders)
- Each situation MUST assign speakers. Match character gender to voice gender.
  Pick character names that fit the scenario.

Available voices:
{voices_json}

Output a JSON array of situations, each with:
- situation_id: "{theme_abbr}_{{domain}}_NNN" format
- theme, domain, scenario, context
- characters: {{ "agent": {{"name": "...", "gender": "male" or "female", "role": "...", "speaking_style": "..."}},
    "human": {{"name": "...", "gender": "male" or "female", "role": "...", "speaking_style": "..."}} }}
  - role: character's job/position (e.g., "customer_service", "doctor", "teacher", "lawyer")
  - speaking_style: how they speak (e.g., "formal and professional", "casual and friendly", "urgent and direct")
- expected_emotion_arc: list of emotions
- difficulty: "easy" | "medium" | "hard"
- key_details: dict of specific values

Theme config:
{theme_json}

Output ONLY the JSON array.
"""

DIALOGUE_PROMPT = """\
Generate a multi-turn spoken dialogue for M-CHAT (Malay + English code-mixed).

Situation:
{situation_json}

Rules:
1. {min_turns}-{max_turns} turns, alternating agent/human
2. Turn 1 MUST be from the agent — agent always opens the conversation (greeting, introduction)
3. Each turn: 1-3 sentences max (TTS-friendly)
4. Natural BM/EN code-mixing (lah, kan, meh, tu, ke, kot)
5. Follow each character's speaking_style from the situation — agent and human may have different registers.
6. Write numbers, amounts, dates, IDs, abbreviations as you normally would.
   Text normalization for TTS is handled separately — just write naturally.
7. Include emotion and tone per turn
8. Follow the expected emotional arc
9. Text must be natural spoken Malay — NOT formal written Malay
10. The LAST turn MUST be from the agent — agent closes the conversation
    (polite closing, summary, farewell, or next-steps). Human never gets the final word.

CRITICAL CHARACTER RULE — The agent is "{agent_name}" ({agent_gender}), the human is "{human_name}" ({human_gender}).
- You MUST use these EXACT names. Do NOT invent, substitute, or change them.
- Malaysian honorifics: Male = "Encik [name]", Female = "Puan [name]" or "Cik [name]"
- Agent {agent_name} ({agent_gender}) uses honorific: {agent_honorific}
- Human {human_name} ({human_gender}) uses honorific: {human_honorific}
- They refer to each other using these honorifics + names
- Third-party NPCs may use any realistic Malaysian name

CRITICAL LANGUAGE RULE — Bahasa Melayu must be the DOMINANT language:
- At least 70% of each turn must be in Bahasa Melayu
- English words are ONLY for: technical terms, brand names,
  proper nouns, short phrases (lah, kan, kot)
- DO NOT write entire sentences or clauses in English — always use BM as the base
- Example BAD: "Can you share more about how the AI works?"
- Example GOOD: "Boleh share lebih lanjut tentang bagaimana AI tu berfungsi?"

Output JSON:
{{{{
  "dialogue_id": "{situation_id}_d01",
  "situation_id": "{situation_id}",
  "turns": [
    {{{{ "turn": 1, "speaker": "agent", "text": "...", "emotion": "...", "tone": "..." }}}}
  ],
  "metadata": {{{{ "total_turns": N, "duration_estimate_s": N,
    "language_breakdown": {{"ms": 0.70, "en": 0.25, "mixed": 0.05}} }}}}
}}}}

Output ONLY valid JSON.
"""


# ── Voice selection ────────────────────────────────────────────────────────────


# ── Stage 1: LLM dialogue generation ──────────────────────────────────────────


async def generate_dialogue(
    situation: dict,
    pipeline_cfg: dict,
    router: LLMRouter,
    output_dir: Path,
    run_suffix: str,
) -> tuple[dict, str] | None:
    """Generate or reuse a dialogue for a situation. Returns (dialogue, llm_backend) or None."""
    sid = situation.get("situation_id", "unknown")
    dialogue_id = f"{sid}_d01{run_suffix}"

    # Reuse existing dialogue from stage3 if available
    dlg_dir = output_dir / "stage3_dialogues"
    dlg_dir.mkdir(parents=True, exist_ok=True)
    existing_path = dlg_dir / f"{sid}.json"
    if existing_path.exists():
        with open(existing_path) as f:
            dialogue = json.load(f)
        dialogue["dialogue_id"] = dialogue_id
        dialogue["situation_id"] = sid
        if not dialogue.get("turns"):
            return None
        logger.debug("Reused existing dialogue for %s", sid)
        return dialogue, dialogue.get("llm_backend", "reused")

    # Extract character info for prompt injection
    chars = situation.get("characters", {})
    agent_char = chars.get("agent", {})
    human_char = chars.get("human", {})
    agent_name = agent_char.get("name", "Agent")
    agent_gender = agent_char.get("gender", "male")
    human_name = human_char.get("name", "User")
    human_gender = human_char.get("gender", "female")
    agent_honorific = f"Encik {agent_name}" if agent_gender == "male" else f"Puan {agent_name}"
    human_honorific = f"Encik {human_name}" if human_gender == "male" else f"Puan {human_name}"

    # Generate new dialogue via LLM
    prompt = DIALOGUE_PROMPT.format(
        situation_json=json.dumps(situation, indent=2, ensure_ascii=False),
        min_turns=pipeline_cfg.get("min_turns", 6),
        max_turns=pipeline_cfg.get("max_turns", 12),
        situation_id=sid,
        agent_name=agent_name,
        agent_gender=agent_gender,
        human_name=human_name,
        human_gender=human_gender,
        agent_honorific=agent_honorific,
        human_honorific=human_honorific,
    )
    try:
        raw, llm_backend = await router.generate(prompt)
        dialogue = parse_json_response(raw)
    except Exception as e:
        logger.error("LLM failed %s: %s", sid, e)
        return None

    # Post-validate character names
    turns = dialogue.get("turns", [])
    if turns:
        all_text = " ".join(t.get("text", "") for t in turns).lower()
        issues = []
        if agent_name.lower() not in all_text:
            issues.append(f"agent name '{agent_name}' not found in dialogue text")
        if human_name.lower() not in all_text:
            issues.append(f"human name '{human_name}' not found in dialogue text")
        if turns[0].get("speaker") != "agent":
            issues.append("turn 1 is not agent (must open)")
        if turns[-1].get("speaker") != "agent":
            issues.append("last turn is not agent (must close)")
        if issues:
            logger.warning("Dialogue %s validation issues: %s", sid, "; ".join(issues))

    dialogue["dialogue_id"] = dialogue_id
    dialogue["situation_id"] = sid
    dialogue["llm_backend"] = llm_backend

    with open(existing_path, "w") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)

    if not dialogue.get("turns"):
        return None
    return dialogue, llm_backend


# ── Stage 2: TTS audio generation ─────────────────────────────────────────────


async def generate_audio(
    dialogue: dict,
    situation: dict,
    pipeline_cfg: dict,
    pool: ServerPool,
    voice_pools: dict[str, list[str]],
    default_genders: dict[str, str],
    output_dir: Path,
    run_suffix: str,
    llm_backend: str,
) -> dict | None:
    """Generate TTS audio for all turns in parallel, combine to stereo."""
    sid = situation.get("situation_id", "unknown")
    dialogue_id = dialogue["dialogue_id"]
    dialogue_dir = output_dir / "stage4_audio" / dialogue_id
    dialogue_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dialogue_dir / "manifest.json"

    if manifest_path.exists():
        return None

    turns = dialogue["turns"]

    # Pick one consistent voice per role from gender pools for this dialogue
    characters = situation.get("characters", {})
    dialogue_voice_map: dict[str, str] = {}
    used_voices: set[str] = set()
    for role, default_gender in default_genders.items():
        char_info = characters.get(role, {})
        gender = char_info.get("gender", default_gender)
        pool_list = voice_pools.get(gender, [])
        available = [v for v in pool_list if v not in used_voices] or pool_list
        voice = random.choice(available) if available else "anwar"
        dialogue_voice_map[role] = voice
        used_voices.add(voice)

    # TTS all turns in parallel
    async def synthesize_turn(turn: dict) -> tuple[dict, tuple[str, np.ndarray]] | None:
        turn_num = turn["turn"]
        speaker = turn["speaker"]
        spoken = preprocess_for_tts(turn["text"])
        voice_id = dialogue_voice_map.get(speaker, "anwar")

        t0 = time.perf_counter()
        try:
            wav_bytes = await pool.synthesize(spoken, voice_id)
        except Exception as e:
            logger.error("TTS failed %s turn %d: %s", sid, turn_num, e)
            return None
        elapsed = time.perf_counter() - t0

        audio = parse_wav_bytes(wav_bytes)
        dur = len(audio) / SAMPLE_RATE
        if dur < 0.1:
            logger.error("Audio too short %s turn %d", sid, turn_num)
            return None

        wav_name = f"turn_{turn_num:02d}_{speaker}.wav"
        write_wav(dialogue_dir / wav_name, audio)

        manifest_entry = {
            "turn": turn_num,
            "speaker": speaker,
            "text_written": turn["text"],
            "text_spoken": spoken,
            "wav_file": wav_name,
            "duration_s": round(dur, 3),
            "tts_time_s": round(elapsed, 3),
            "voice_id": voice_id,
        }
        return manifest_entry, (speaker, apply_fade(audio))

    turn_results = await asyncio.gather(*[synthesize_turn(t) for t in turns])

    # Check for failures
    if any(r is None for r in turn_results):
        return None

    manifest_turns = [r[0] for r in turn_results]
    turn_audios = [r[1] for r in turn_results]

    # Validate turn counts
    if len(turn_audios) != len(turns):
        logger.error(
            "Turn count mismatch: expected %d, got %d for %s",
            len(turns), len(turn_audios), sid,
        )
        return None

    # Stereo + MP3
    gap_samples = int(pipeline_cfg.get("gap_between_turns_s", 0.3) * SAMPLE_RATE)
    stereo = build_stereo(turn_audios, gap_samples)
    write_wav_stereo(dialogue_dir / "combined.wav", stereo)

    mp3_bitrate = pipeline_cfg.get("mp3_bitrate", "192k")
    wav_to_mp3(dialogue_dir / "combined.wav", dialogue_dir / "combined.mp3", mp3_bitrate)

    total_dur = stereo.shape[1] / SAMPLE_RATE
    manifest = {
        "dialogue_id": dialogue_id,
        "llm_backend": llm_backend,
        "situation": {
            "situation_id": sid,
            "theme": situation.get("theme", ""),
            "domain": situation.get("domain", ""),
            "scenario": situation.get("scenario", ""),
            "context": situation.get("context", ""),
            "characters": situation.get("characters", {}),
            "expected_emotion_arc": situation.get("expected_emotion_arc", []),
            "difficulty": situation.get("difficulty", ""),
            "key_details": situation.get("key_details", {}),
        },
        "turns": manifest_turns,
        "combined_wav": "combined.wav",
        "combined_mp3": "combined.mp3",
        "total_duration_s": round(total_dur, 3),
        "format": "stereo",
        "channels": {"left": "agent", "right": "human"},
        "sample_rate": SAMPLE_RATE,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tts_total = sum(t["tts_time_s"] for t in manifest_turns)
    logger.info(
        "%s: %d turns, %.1fs audio, %.1fs TTS",
        sid, len(turns), total_dur, tts_total,
    )
    return manifest


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


async def run_pipeline(
    config: dict, max_concurrent: int,
    max_dialogues: int = 0, continuous: bool = False,
) -> None:
    output_dir = Path(config["output_dir"])
    pipeline_cfg = config["pipeline"]
    tts_cfg = config["tts"]

    # ── Situation validation ────────────────────────────────────────────────
    SERVICE_KEYWORDS = {"agent", "officer", "staff", "service", "advisor", "consultant",
                        "mechanic", "trainer", "teacher", "doctor", "nurse", "receptionist",
                        "planner", "counselor", "host", "dj", "moderator", "teller",
                        "pharmacist", "therapist", "instructor", "paramedic", "dispatcher"}
    CUSTOMER_KEYWORDS = {"customer", "client", "patient", "student", "guest", "applicant",
                         "buyer", "member", "owner", "caller", "parent", "bride", "groom",
                         "passenger", "tenant", "policyholder", "listener", "viewer"}
    TTS_VOICE_NAMES_UNDER = {"Pearl_Happy", "Pearl_Sad", "Pearl_Angry1", "Pearl_Angry2",
                              "Pearl_Surprise", "hendrick-6", "wafiy_5", "Farid_595", "Dania_0"}
    TTS_VOICE_NAMES = TTS_VOICE_NAMES_UNDER | {n.replace("_", " ") for n in TTS_VOICE_NAMES_UNDER}
    MALAYSIAN_NAMES_MALE = ["Ahmad", "Hafiz", "Amir", "Rizal", "Faiz", "Zul", "Idris",
                             "Nazri", "Syafiq", "Kamal", "Zainal", "Rashid", "Fahmi",
                             "Aizat", "Nizam", "Syahmi", "Hazim", "Amirul", "Danial",
                             "Fikri", "Hakim", "Irfan", "Luqman", "Mukhriz"]
    MALAYSIAN_NAMES_FEMALE = ["Siti", "Aisyah", "Nurul", "Fatimah", "Zara", "Amira",
                               "Hani", "Diana", "Farah", "Lina", "Nadia", "Salina",
                               "Puteri", "Nabila", "Syazwani", "Noor", "Aida", "Raihana",
                               "Surya", "Balqis", "Izzah", "Mira", "Syafiqah", "Zuleyka"]
    _used_name_ids: set[str] = set()

    def validate_situation(sit: dict) -> dict:
        """Validate + fix situation: swap agent/human if roles are reversed, fix TTS names."""
        chars = sit.get("characters", {})
        agent = chars.get("agent", {})
        human = chars.get("human", {})

        # Check if roles are swapped (human is service provider, agent is customer)
        agent_role = agent.get("role", "").lower()
        human_role = human.get("role", "").lower()
        agent_is_service = any(kw in agent_role for kw in SERVICE_KEYWORDS)
        human_is_service = any(kw in human_role for kw in SERVICE_KEYWORDS)
        agent_is_customer = any(kw in agent_role for kw in CUSTOMER_KEYWORDS)

        if human_is_service and (agent_is_customer or not agent_is_service):
            chars["agent"], chars["human"] = chars["human"], chars["agent"]
            logger.info("Swapped agent↔human in %s: agent=%s→%s, human=%s→%s",
                        sit.get("situation_id"), agent_role, human_role, human_role, agent_role)

        # Fix TTS voice names as character names
        sid = sit.get("situation_id", "")
        for role_key in ("agent", "human"):
            name = chars.get(role_key, {}).get("name", "")
            if name in TTS_VOICE_NAMES:
                gender = chars[role_key].get("gender", "male")
                pool = MALAYSIAN_NAMES_MALE if gender == "male" else MALAYSIAN_NAMES_FEMALE
                # Pick a unique name per situation+role to avoid duplicates
                for candidate in pool:
                    name_id = f"{sid}:{role_key}:{candidate}"
                    if name_id not in _used_name_ids:
                        _used_name_ids.add(name_id)
                        chars[role_key]["name"] = candidate
                        break
                else:
                    chars[role_key]["name"] = pool[0]
                logger.info("Fixed TTS name in %s %s: %s→%s",
                            sid, role_key, name, chars[role_key]["name"])

        sit["characters"] = chars
        return sit

    # Load themes
    themes_path = output_dir / "stage1_themes.json"
    if not themes_path.exists():
        logger.error("Run 01_generate_themes.py first")
        sys.exit(1)
    with open(themes_path) as f:
        themes = json.load(f)

    # Init
    servers = tts_cfg.get("servers", [{"url": "http://localhost:8880", "max_concurrent": 4}])
    pool = ServerPool(servers, trim_seconds=tts_cfg.get("trim_front_seconds", 0.0))

    backends = config.get("llm_backends")
    if backends:
        router = LLMRouter(backends)
    else:
        router = LLMRouter([config["llm"]])
    llm_primary = router.get_primary()
    voice_pools = tts_cfg.get("voice_pools", {})
    default_genders = tts_cfg.get("default_genders", {"agent": "male", "human": "female"})

    # Fetch available voices from TTS server (context for situation generation)
    voices_json = "[]"
    try:
        async with httpx.AsyncClient(base_url=servers[0]["url"]) as c:
            resp = await c.get("/voices", timeout=10.0)
            resp.raise_for_status()
            voices_data = resp.json().get("voices", [])
            voices_json = json.dumps(voices_data, indent=2, ensure_ascii=False)
            logger.info("Fetched %d voices from TTS server", len(voices_data))
    except Exception as e:
        logger.warning(
            "Failed to fetch voices from TTS server: %s", e)

    # Two-stage pipeline: situations → dialogues (LLM) → audio (TTS)
    n_llm_workers = min(max_concurrent, 4)
    n_tts_workers = max_concurrent
    SitItem = tuple[dict, str] | None
    DlgItem = tuple[dict, dict, str, str] | None
    situation_queue: asyncio.Queue[SitItem] = asyncio.Queue(maxsize=n_llm_workers * 2)
    dialogue_queue: asyncio.Queue[DlgItem] = asyncio.Queue(
        maxsize=n_tts_workers * 2,
    )
    results: list[dict | Exception | None] = []
    total_situations = 0
    logger.info("Workers: %d LLM, %d TTS", n_llm_workers, n_tts_workers)

    # ── Stage 0: Produce situations (loops forever in continuous mode) ────

    async def produce_batch(run_suffix: str) -> int:
        """Generate situations for all themes. Returns count of new situations."""
        nonlocal total_situations
        batch_count = 0

        sit_dir = output_dir / "stage2_situations"
        sit_dir.mkdir(parents=True, exist_ok=True)

        # Load existing situations from disk (resumable)
        for sf in sorted(sit_dir.glob("*.json")):
            if sf.name.startswith("_"):
                continue
            with open(sf) as f:
                try:
                    for sit in json.load(f):
                        sit = validate_situation(sit)
                        total_situations += 1
                        batch_count += 1
                        await situation_queue.put((sit, run_suffix))
                        if max_dialogues > 0 and total_situations >= max_dialogues:
                            return batch_count
                except Exception as e:
                    logger.warning("Failed to load %s: %s", sf.name, e)

        if batch_count > 0:
            logger.info("Loaded %d existing situations", batch_count)

        # Generate new situations per theme
        for theme in themes:
            if max_dialogues > 0 and total_situations >= max_dialogues:
                break
            theme_name = theme["theme"]
            theme_file = sit_dir / f"{theme_name}.json"
            if theme_file.exists():
                continue

            n = pipeline_cfg.get("situations_per_theme", 20)
            seed = random.randint(10000, 99999)
            prompt = SITUATION_PROMPT.format(
                n=n,
                seed=seed,
                theme_abbr=theme_name[:2],
                theme_json=json.dumps(theme, indent=2, ensure_ascii=False),
                voices_json=voices_json,
            )
            try:
                raw = await llm_primary.generate(prompt)
                situations = parse_json_response(raw)
                logger.info("Generated %d situations for '%s'", len(situations), theme_name)
            except Exception as e:
                logger.error("Situation generation failed for '%s': %s", theme_name, e)
                fail_file = sit_dir / f"_failed_{theme_name}.json"
                fail_entry = {
                    "theme": theme_name,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "prompt_snippet": prompt[:500],
                }
                failures = []
                if fail_file.exists():
                    try:
                        with open(fail_file) as ff:
                            failures = json.load(ff)
                    except Exception:
                        pass
                failures.append(fail_entry)
                with open(fail_file, "w") as ff:
                    json.dump(failures, ff, indent=2, ensure_ascii=False)
                continue

            if situations:
                situations = [validate_situation(s) for s in situations]
                with open(theme_file, "w") as f:
                    json.dump(situations, f, indent=2, ensure_ascii=False)
                for sit in situations:
                    total_situations += 1
                    batch_count += 1
                    await situation_queue.put((sit, run_suffix))
                    if max_dialogues > 0 and total_situations >= max_dialogues:
                        break

        return batch_count

    async def producer() -> None:
        if continuous:
            batch_num = 0
            while True:
                batch_num += 1
                run_suffix = f"_{random.randint(1000, 9999)}"
                logger.info("=== Continuous batch %d (suffix %s) ===", batch_num, run_suffix)

                # Clean stage2+3 so fresh situations are generated
                for p in (output_dir / "stage2_situations", output_dir / "stage3_dialogues"):
                    if p.exists():
                        for f in p.glob("*.json"):
                            if not f.name.startswith("_"):
                                f.unlink()

                count = await produce_batch(run_suffix)
                logger.info("Batch %d: enqueued %d situations", batch_num, count)
        else:
            run_suffix = f"_{random.randint(1000, 9999)}"
            logger.info("Run suffix: %s (ensures unique dialogue IDs)", run_suffix)
            count = await produce_batch(run_suffix)
            logger.info("Enqueued %d situations total", count)
            for _ in range(n_llm_workers):
                await situation_queue.put(None)

    # ── Stage 1: LLM dialogue generation ───────────────────────────────────

    async def llm_worker(worker_id: int) -> None:
        while True:
            item = await situation_queue.get()
            if item is None:
                break
            sit, run_suffix = item
            try:
                result = await generate_dialogue(
                    sit, pipeline_cfg, router, output_dir, run_suffix,
                )
                if result is None:
                    results.append(None)
                else:
                    dialogue, llm_backend = result
                    await dialogue_queue.put((dialogue, sit, llm_backend, run_suffix))
            except Exception as e:
                logger.error(
                    "LLM worker %d failed on %s: %s",
                    worker_id, sit.get("situation_id"), e,
                )
                fail_dir = output_dir / "stage3_dialogues"
                fail_dir.mkdir(parents=True, exist_ok=True)
                fail_file = fail_dir / "_failed_dialogues.jsonl"
                with open(fail_file, "a") as ff:
                    json.dump({
                        "situation_id": sit.get("situation_id"),
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "situation_snippet": json.dumps(sit, ensure_ascii=False)[:300],
                    }, ff, ensure_ascii=False)
                    ff.write("\n")
                results.append(e)

        if not continuous:
            for _ in range(n_tts_workers // n_llm_workers + 1):
                await dialogue_queue.put(None)

    # ── Stage 2: TTS audio generation ──────────────────────────────────────

    tts_done = 0

    async def tts_worker(worker_id: int) -> None:
        nonlocal tts_done
        while True:
            item = await dialogue_queue.get()
            if item is None:
                tts_done += 1
                if tts_done >= n_llm_workers:
                    for _ in range(n_tts_workers - 1):
                        await dialogue_queue.put(None)
                break
            dialogue, sit, llm_backend, run_suffix = item
            try:
                result = await generate_audio(
                    dialogue, sit, pipeline_cfg, pool,
                    voice_pools, default_genders, output_dir,
                    run_suffix, llm_backend,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "TTS worker %d failed on %s: %s",
                    worker_id, sit.get("situation_id"), e,
                )
                results.append(e)

    t_total = time.perf_counter()

    prod_task = asyncio.create_task(producer())
    llm_tasks = [asyncio.create_task(llm_worker(i)) for i in range(n_llm_workers)]
    tts_tasks = [asyncio.create_task(tts_worker(i)) for i in range(n_tts_workers)]

    if continuous:
        # In continuous mode, we never naturally finish — run until cancelled
        await asyncio.gather(prod_task, *llm_tasks, *tts_tasks)
    else:
        await prod_task
        await asyncio.gather(*llm_tasks)
        await asyncio.gather(*tts_tasks)

    await pool.aclose()
    await router.aclose()

    elapsed = time.perf_counter() - t_total
    generated = sum(1 for r in results if isinstance(r, dict))
    skipped = sum(1 for r in results if r is None)
    errors = sum(1 for r in results if isinstance(r, Exception))

    logger.info(
        "Pipeline complete: %d situations → %d dialogues generated, "
        "%d skipped, %d errors (%.1fs)",
        total_situations, generated, skipped, errors, elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="M-CHAT Async Pipeline")
    parser.add_argument(
        "--max-concurrent", type=int, default=8,
        help="Max concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--max-dialogues", type=int, default=0,
        help="Max dialogues to generate (0 = unlimited)",
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously — never stop generating batches",
    )
    args = parser.parse_args()

    config = load_config()
    asyncio.run(run_pipeline(
        config, args.max_concurrent,
        max_dialogues=args.max_dialogues, continuous=args.continuous,
    ))


if __name__ == "__main__":
    main()
