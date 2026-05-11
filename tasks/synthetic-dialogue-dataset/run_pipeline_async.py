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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)
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


def trim_leading_silence(audio: np.ndarray, threshold: float = 0.01, min_silence: int = 512) -> np.ndarray:
    """Trim leading silence from TTS audio. Keeps at least min_silence samples."""
    if len(audio) <= min_silence:
        return audio
    abs_audio = np.abs(audio)
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return audio[-min_silence:]
    start = max(0, above[0] - 160)  # keep 10ms of context before first sound
    return audio[start:]


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
        self._sem = asyncio.Semaphore(1)

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
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._sem = asyncio.Semaphore(32)

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
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    # Strip <think/> blocks from Qwen3 reasoning
                    text = re.sub(r"<think[\s\S]*?</think\s*>", "", text).strip()
                    return text
                except Exception as e:
                    logger.warning("[%s] attempt %d: %s: %s", self.name, attempt + 1, type(e).__name__, e)
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
        """Race all backends concurrently, return first success."""
        winner: asyncio.Future[tuple[str, str]] = asyncio.get_running_loop().create_future()

        async def _try(client: LLMClient | VLLMClient) -> None:
            try:
                text = await client.generate(prompt)
                if not winner.done():
                    winner.set_result((text, client.name))
            except Exception as e:
                if not winner.done():
                    logger.debug("[%s] lost race: %s", client.name, e)

        tasks = [asyncio.create_task(_try(c)) for c in self._clients]
        try:
            return await winner
        finally:
            for t in tasks:
                t.cancel()

    async def aclose(self) -> None:
        for c in self._clients:
            if hasattr(c, "aclose"):
                await c.aclose()


def parse_json_response(text: str) -> dict | list:
    from json_repair import repair_json
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(text)
            parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
            logger.debug("JSON repaired successfully")
        except Exception:
            match = re.search(r'\[[\s\S]*\]', text)
            if not match:
                match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    repaired = repair_json(match.group(0))
                    parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
                    logger.debug("JSON extracted + repaired via regex")
                except Exception as e:
                    raise ValueError(f"JSON repair failed: {e}") from e
            else:
                raise
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
- context_details: {{ "location": "...", "environment": "...",
  "circumstances": "...", "time_of_day": "..." }}
  - location: where it takes place (e.g., "Maybank branch PJ",
    "clinic lobby", "food court")
  - environment: channel/medium (e.g., "phone call", "video call", "chat app", "voice assistant")
    IMPORTANT: All conversations are remote (phone/video/chat). Never use "face-to-face". No physical actions (sitting, pointing, handing items).
  - circumstances: what led to this conversation (e.g., "customer noticed unauthorized transaction",
    "patient has recurring headache for 2 weeks")
  - time_of_day: e.g., "morning rush hour", "afternoon", "late night"
- characters: {{ "agent": {{...}}, "human": {{...}} }} — each character has:
  - name, gender, role, speaking_style (as before)
  - personality_traits: 2-3 traits (e.g., "patient and methodical", "anxious and verbose",
    "cheerful and efficient")
  - background: one sentence about who they are (e.g., "Senior customer service rep with 5 years
    experience", "First-time home buyer in their late 20s")
  - emotional_state: their initial mood (e.g., "calm and professional", "frustrated", "excited",
    "nervous")
  - goals: what they want to achieve (e.g., "resolve the complaint quickly",
    "understand the medical procedure", "get the best deal possible")
  - communication_quirks: unique speech habit (e.g., "uses 'kan' a lot", "very formal even in
    casual situations", "frequently code-switches mid-sentence", "uses Malaysian slang like 'kot',
    'lah', 'meh'")
- expected_emotion_arc: list of emotions
- difficulty: "easy" | "medium" | "hard"
- key_details: dict of specific values

Theme config:
{theme_json}

Output ONLY the JSON array.
"""

DIALOGUE_PROMPT = """\
Generate a multi-turn spoken dialogue for M-CHAT (Malay + English code-mixed).

This is for a SPEECH dataset — the text will be converted to audio and used to train
AI models. Every turn must sound like a REAL Malaysian speaking naturally on a phone call.
NOT scripted, NOT formal, NOT robotic — like overhearing a real conversation at a kopitiam.

Situation:
{situation_json}

Rules:
1. {min_turns}-{max_turns} turns, alternating agent/human
2. Turn 1 MUST be from the agent — agent always opens the conversation (greeting, introduction)
3. Each turn: 1-3 sentences max (TTS-friendly)
4. Natural BM/EN code-mixing (lah, kan, meh, tu, ke, kot, pun, je, keje)
5. Follow each character's speaking_style AND personality_traits from the situation.
   Embody their personality — a "patient" agent speaks differently from a "brusque" one.
6. Use each character's communication_quirks naturally (e.g., if they use "kan" a lot,
   include it in their dialogue; if they code-switch frequently, reflect that).
7. The conversation should reflect the context_details (location, environment,
   circumstances, time_of_day). A late-night call feels different from a morning
   call. All conversations are remote — no physical actions or face-to-face settings.
8. Each character pursues their stated goals — this creates natural tension and
   resolution in the dialogue.
9. Write numbers, amounts, dates, IDs, abbreviations as you normally would.
   Text normalization for TTS is handled separately — just write naturally.
10. Include emotion and tone per turn
11. Follow the expected emotional arc
12. Text must be NATURAL SPOKEN Malay — like how Malaysians ACTUALLY talk:
    - "Saya nak tanya" not "Saya ingin bertanya"
    - "Dah lama ke ni?" not "Sudah berapa lama?"
    - "Alamak, serious ke ni" not "Oh tidak, adakah ini serius?"
    - Include slang, truncations, contractions: "nak", "dah", "tak", "je", "keje", "kot"
    - Use exclamation and emotion naturally: "Wah!", "Alamak!", "Hah?", "Eh?", "Oo"
13. The LAST turn MUST be from the agent — agent closes the conversation
    (polite closing, summary, farewell, or next-steps). Human never gets the final word.
14. Never include physical actions (sit down, come here, take this, point at). All
    conversations are phone calls, video calls, or chat — no face-to-face interaction.
15. ANTI-REPETITION (CRITICAL): Each turn MUST advance the conversation. NEVER repeat
    the same phrase, sentence, or acknowledgment pattern. If you wrote "saya faham" once,
    don't write it again — use a different phrase. Each response must introduce NEW information,
    ask a NEW question, or react to something specific the other person just said. Stalling
    phrases like "baiklah", "saya faham", "jangan risau" should appear at most ONCE per dialogue.

NATURAL ACKNOWLEDGMENT VOCABULARY (CRITICAL — use DIFFERENT one each time):
When acknowledging what the other person said, pick from this RICH vocabulary.
NEVER use the same acknowledgment twice in one dialogue. Each pick must be DIFFERENT.

  Understanding: "Oo, macam tu ke", "Alamak", "Wah", "Betul tu", "Setuju",
    "Oo, saya tengok", "Macam tu lah", "Noted", "Boleh", "Ha, macam tu",
    "Oo, baiklah", "Haah, saya dengar", "Ya Allah", "Wah, serious ke",
    "Hmmm, faham", "Tapi tu lah", "Ye ke?", "Tak pe", "Ok, noted"
  Agreement: "Setuju", "Betul tu", "Memang lah", "Ya lah", "Punya lah",
    "Mestilah", "Baru betul tu", "Tu lah masalahnya", "Ha, betul"
  Surprise: "Wah!", "Alamak!", "Hah?", "Ye ke?", "Serious?", "Masyaallah",
    "Astagfirullah", "Teruk gila", "Gila ke?", "Amboi"
  Transition: "Macam mana pulak", "Tapi...", "Okay, tapi...", "Apa kata...",
    "Bila punya hal ni", "Jadi macam mana", "Habis tu"
  Closing: "Okay, saya catat", "Nanti saya follow up",
    "Saya akan check", "Boleh, saya uruskan", "Nanti saya email",
    "Kita proceed macam ni", "Saya bagi update esok",
    "Okay, settle", "Macam tu dah", "Noted, saya tengok dulu",
    "Boleh je", "Saya ambil nota", "Nanti saya hubungi balik"

FORBIDDEN — do NOT use these robotic crutch phrases more than ONCE in the entire dialogue:
  "Saya faham", "Jangan risau", "Tidak apa", "Baiklah, saya faham", "Baiklah"
  These sound like a chatbot, not a human. Use the natural vocabulary above instead.
20. CHARACTER NAME AWARENESS — Do NOT confuse character names with voice names.
    The agent's name is "{agent_name}", NOT the voice model name. The human's
    name is "{human_name}", NOT the voice model name. Never refer to yourself
    by the voice model name or use the other character's name for yourself.
    If the situation describes "Siti Nora" as the agent, use "Siti Nora" in dialogue,
    not a different name.

CRITICAL CHARACTER RULE — The agent is "{agent_name}" ({agent_gender}), the human is "{human_name}" ({human_gender}).
- You MUST use these EXACT names in the dialogue text. Do NOT invent, substitute, or change them.
- NEVER use generic labels like "Agent", "User", "Caller", "Customer", "Puan User", "Encik Agent".
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



QUALITY_PROMPT = """\
Score this Malay spoken dialogue on 4 dimensions (1-5 each).

Situation context: {context_summary}

Dialogue:
{dialogue_text}

Rate each dimension:
1. coherence: Do turns logically follow each other? No contradictions or non-sequiturs?
2. naturalness: Does this sound like real spoken conversation? No robotic/forced phrasing?
3. persona_consistency: Do characters stay true to their personality, speaking style, and emotional state?
4. topic_coverage: Does the dialogue fully explore the situation with concrete details?

Output ONLY JSON:
{{"coherence": N, "naturalness": N, "persona_consistency": N, "topic_coverage": N, "overall": N, "reason": "one sentence"}}
"""


async def score_dialogue(
    dialogue: dict,
    situation: dict,
    router: LLMRouter,
) -> dict | None:
    """LLM-as-judge quality scoring. Returns scores dict or None on failure."""
    chars = situation.get("characters", {})
    agent_char = chars.get("agent", {})
    human_char = chars.get("human", {})
    ctx = situation.get("context_details", {})

    context_summary = (
        f"Theme: {situation.get('theme', 'unknown')}. "
        f"Scenario: {situation.get('scenario', 'unknown')}. "
        f"Agent: {agent_char.get('name', 'Agent')} ({agent_char.get('personality_traits', 'unknown')}), "
        f"Human: {human_char.get('name', 'Human')} ({human_char.get('personality_traits', 'unknown')}). "
        f"Context: {ctx.get('location', 'unknown')}, {ctx.get('environment', 'unknown')}, "
        f"{ctx.get('circumstances', 'unknown')}"
    )

    turns = dialogue.get("turns", [])
    dialogue_text = "\n".join(
        f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns
    )

    prompt = QUALITY_PROMPT.format(
        context_summary=context_summary,
        dialogue_text=dialogue_text,
    )
    try:
        raw, backend_name = await router.generate(prompt)
        scores = parse_json_response(raw)
        return scores
    except Exception as e:
        logger.warning("Quality scoring failed: %s", e)
        return None


# ── Voice selection ────────────────────────────────────────────────────────────


_FIX_PROMPT = """Fix this Malaysian dialogue turn. The agent said: "{turn_text}"
It contains the robotic phrase "{phrase}" which is repeated elsewhere in the dialogue.
Rewrite ONLY the agent's line to sound natural — like a real Malaysian speaking, not a chatbot.

Context (previous turns):
{context}

Natural alternatives: "Oo, macam tu ke", "Betul tu", "Wah", "Setuju", "Memang lah", "Ya lah",
"Ha, okay", "Alamak", "Noted", "Boleh je", "Macam tu dah", "Saya tengok dulu",
"Nanti saya follow up", "Okay, settle", "Saya ambil nota"

Reply with ONLY the rewritten agent turn. No quotes, no speaker label, just the Malay text."""


async def _fix_boring_turns(
    turns: list[dict], router: LLMRouter, sid: str, max_attempts: int = 2
) -> list[dict] | None:
    """Rewrite agent turns that repeat boring phrases, keeping the first occurrence."""
    _boring = ["saya faham", "jangan risau", "saya faham,", "jangan risau,", "baiklah,", "tak apa,"]

    for _attempt in range(max_attempts):
        fixes_applied = False
        for phrase in _boring:
            agent_indices = [
                i for i, t in enumerate(turns)
                if t.get("speaker") == "agent" and phrase in t.get("text", "").lower()
            ]
            if len(agent_indices) < 2:
                continue
            # Keep first occurrence, fix the rest
            for idx in agent_indices[1:]:
                prev = turns[max(0, idx - 2):idx]
                context = "\n".join(f"  {t['speaker']}: {t['text']}" for t in prev)
                prompt = _FIX_PROMPT.format(
                    turn_text=turns[idx]["text"], phrase=phrase, context=context or "  (start of dialogue)"
                )
                try:
                    fixed, _ = await router.generate(prompt)
                    fixed = fixed.strip().strip('"').strip("'")
                    if fixed and phrase not in fixed.lower():
                        logger.info("Fixed turn %d in %s: replaced '%s' with '%s'",
                                    idx, sid, turns[idx]["text"][:50], fixed[:50])
                        turns[idx]["text"] = fixed
                        fixes_applied = True
                except Exception as e:
                    logger.warning("Fix failed for turn %d in %s: %s", idx, sid, e)

        if not fixes_applied:
            break

    # Final check — reject if still broken
    agent_lower = [t.get("text", "").lower() for t in turns if t.get("speaker") == "agent"]
    for phrase in _boring:
        if sum(1 for at in agent_lower if phrase in at) >= 2:
            return None
    return turns


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

    # Reject malformed dialogues
    if not dialogue.get("turns"):
        logger.error("Dialogue %s rejected: no turns", sid)
        return None

    # Fix empty scenario by constructing from context_details
    if not situation.get("scenario"):
        ctx = situation.get("context_details", {})
        chars_info = situation.get("characters", {})
        agent_role = chars_info.get("agent", {}).get("role", "agent")
        human_role = chars_info.get("human", {}).get("role", "caller")
        loc = ctx.get("location", "unknown location")
        circ = ctx.get("circumstances", "routine call")
        situation["scenario"] = f"{human_role} calling {agent_role} at {loc} — {circ}"
        logger.warning("Dialogue %s: constructed missing scenario from context", sid)

    # Fix voice IDs used as character names (replace with agent/human name)
    voice_id_names = {"Pearl Happy", "Pearl Sad", "Pearl Angry1", "Pearl Angry2",
                      "Pearl Surprise", "Pearl_Happy", "Pearl_Sad", "Pearl_Angry1",
                      "Pearl_Angry2", "Pearl_Surprise", "hendrick-6", "wafiy_5",
                      "Farid_595", "Dania_0", "NorinaYahya", "SitiNora", "Aisyah", "Sofea"}
    for role in ["agent", "human"]:
        cname = chars.get(role, {}).get("name", "")
        if cname in voice_id_names:
            chars[role]["name"] = agent_name if role == "agent" else human_name
            logger.warning("Dialogue %s: replaced voice ID '%s' in %s with '%s'",
                           sid, cname, role, chars[role]["name"])
            situation["characters"] = chars

    # Fix generic placeholder names (replace with agent/human name)
    generic_names = {"Agent", "agent", "User", "user", "Caller", "caller",
                     "Customer", "customer", "Encik Agent", "Puan Agent",
                     "Encik User", "Puan User"}
    for role in ["agent", "human"]:
        cname = chars.get(role, {}).get("name", "")
        if cname in generic_names or not cname.strip():
            chars[role]["name"] = agent_name if role == "agent" else human_name
            logger.warning("Dialogue %s: replaced generic %s name '%s' with '%s'",
                           sid, role, cname, chars[role]["name"])
            situation["characters"] = chars

    # Fix same-name collision (both sides same name)
    agent_char_name = chars.get("agent", {}).get("name", agent_name)
    human_char_name = chars.get("human", {}).get("name", human_name)
    if agent_char_name and human_char_name and agent_char_name == human_char_name:
        chars["human"]["name"] = human_name
        logger.warning("Dialogue %s: same-name collision '%s', replaced human with '%s'",
                       sid, agent_char_name, human_name)
        situation["characters"] = chars

    # Fix missing theme by deriving from situation_id prefix
    _theme_prefix_map = {
        "cu": "customer_support", "fo": "food_dining", "tr": "transport",
        "re": "retail", "go": "government_services", "wo": "workplace",
        "rs": "real_estate", "bu": "business", "fi": "finance",
        "fa": "family", "nb": "neighbourhood", "rl": "religious",
        "sp": "sports_leisure", "md": "medical", "mh": "mental_health",
        "wl": "wellness", "ed": "education", "sk": "skills_training",
        "po": "podcast", "ra": "radio_show", "cc": "content_creation",
        "em": "emergency", "le": "legal", "dd": "debate_discussion",
        "ho": "hotel_hospitality", "in": "insurance", "au": "auto_mechanic",
        "im": "immigration", "pa": "property_agent", "tc": "tuition_center",
        "wp": "wedding_planning", "fg": "fitness_gym", "qs": "quiz_show",
        "oc": "open_conversation", "ff": "fun_facts", "en": "entertainment",
        "fc": "foodie_chat", "ts": "travel_stories", "dl": "daily_life",
        "fp": "festival_prep", "gh": "ghost_stories", "sc": "scam_awareness",
        "os": "online_shopping", "wd": "weather_disaster", "pc": "pet_care",
        "lx": "language_exchange", "hm": "home_maintenance", "sr": "school_parent",
        "ra": "relationship_advice", "go": "gossip", "dr": "driving_stories",
    }
    if not situation.get("theme"):
        prefix = sid.split("_")[0][:2] if "_" in sid else sid[:2]
        derived = _theme_prefix_map.get(prefix)
        if derived:
            situation["theme"] = derived
            logger.warning("Dialogue %s: derived missing theme='%s' from sid prefix '%s'", sid, derived, prefix)
        else:
            logger.error("Dialogue %s rejected: missing theme (no prefix match)", sid)
            return None

    # Post-validate character names & clean speaker prefixes from text
    turns = dialogue.get("turns", [])

    # Reject off-topic dialogues (theme/content mismatch)
    theme = situation.get("theme", "").lower()
    scenario = situation.get("scenario", "").lower()
    domain = situation.get("domain", "").lower()
    all_turn_text = " ".join(t.get("text", "") for t in turns).lower()
    _domain_keywords = {
        "banking": ["maybank", "cimb", "bank islam", "rhb bank", "akaun bank", "transaksi", "pinjaman", "atm", "kad debit"],
        "medical": ["doktor", "klinik", "hospital", "sakit", "ubat", "rawatan", "demam", "x-ray"],
        "insurance": ["insuran", "polisi", "takaful", "klaim", "coverage", "premis"],
        "education": ["sekolah", "universiti", "pelajar", "guru", "subjek", "peperiksaan", "kursus"],
        "government": ["jabatan", "kementerian", "kerajaan", "permohonan", "surat beran", "mykad"],
        "immigration": ["passport", "imigresen", "visa", "work permit", "permanent residence"],
        "retail": ["kedai", "beli", "harga", "diskaun", "promo", "pulangkan"],
        "food": ["makan", "restoran", "menu", "makanan", "minuman", "masak"],
        "transport": ["bas", "train", "flight", "grab", "teksi", "tiket", "lapangan terbang"],
        "real_estate": ["rumah", "sewa", "condo", "apartment", "deposit", "tanah"],
    }
    _theme_domain_aliases = {
        "banking": {"finance", "insurance", "business", "real_estate", "property_agent"},
        "medical": {"mental_health", "wellness", "fitness_gym", "insurance"},
        "education": {"skills_training", "tuition_center", "debate_discussion", "quiz_show",
                      "school_parent", "language_exchange"},
        "food": {"food_dining", "foodie_chat", "wedding_planning", "hotel_hospitality",
                 "casual_chat", "daily_life", "open_conversation", "entertainment",
                 "content_creation", "podcast", "radio_show", "fun_facts", "family",
                 "neighbourhood", "festival_prep", "wellness"},
        "retail": {"customer_support", "auto_mechanic", "business", "online_shopping", "wedding_planning"},
        "real_estate": {"property_agent", "hotel_hospitality", "family"},
        "transport": {"travel_stories", "emergency"},
        "government": {"government_services", "immigration", "legal"},
        "insurance": {"finance", "business"},
    }
    dialogue_domain_hits = {}
    for d, keywords in _domain_keywords.items():
        hits = sum(1 for kw in keywords if kw in all_turn_text)
        if hits >= 2:
            dialogue_domain_hits[d] = hits
    if dialogue_domain_hits:
        top_domain = max(dialogue_domain_hits, key=dialogue_domain_hits.get)
        top_hits = dialogue_domain_hits[top_domain]
        theme_domain = f"{theme} {domain} {scenario}"
        aliases = _theme_domain_aliases.get(top_domain, set())
        domain_relevant = (top_domain in theme_domain or
                          theme in aliases or
                          any(kw in theme_domain for kw in _domain_keywords.get(top_domain, [])))
        if not domain_relevant and top_hits >= 3:
            logger.error("Dialogue %s rejected: theme/content mismatch — theme=%s but dialogue is about %s (%d keyword hits)",
                         sid, theme, top_domain, top_hits)
            return None
    voice_bad_names = {"Pearl Happy", "Pearl Sad", "Pearl Angry1", "Pearl Angry2",
                       "Pearl Surprise", "Pearl_Happy", "Pearl_Sad", "Pearl_Angry1",
                       "Pearl_Angry2", "Pearl_Surprise", "hendrick-6", "wafiy_5",
                       "Farid_595", "Dania_0", "Aisyah", "Sofea"}
    for t in turns:
        # Fix qwen3 sometimes using "user" instead of "human"
        if t.get("speaker") == "user":
            t["speaker"] = "human"
        txt = t.get("text", "")
        # Strip LLM-generated speaker prefixes like "Puan Dania:", "Encik Farid:", etc.
        if ":" in txt[:40]:
            txt = txt.split(":", 1)[1].strip()
        # Replace voice names with proper character names
        for bad in voice_bad_names:
            if bad in txt:
                spk = t.get("speaker", "agent")
                if spk == "agent":
                    txt = txt.replace(bad, agent_name)
                else:
                    txt = txt.replace(bad, human_name)
        t["text"] = txt
        # Replace generic labels in turn text
        generic_labels = ["Puan User", "Encik User", "Puan Agent", "Encik Agent",
                         "Encik Caller", "Puan Caller", "Encik Customer", "Puan Customer"]
        spk = t.get("speaker", "agent")
        for gl in generic_labels:
            if gl in txt:
                txt = txt.replace(gl, agent_honorific if spk == "agent" else human_honorific)
                logger.warning("Dialogue %s: replaced generic label '%s' in text", sid, gl)
    if turns:
        # Fix same-speaker ending: strip trailing same-speaker turns
        while len(turns) >= 2 and turns[-1]["speaker"] == turns[-2]["speaker"]:
            removed = turns.pop()
            logger.warning("Dialogue %s: removed trailing same-speaker turn (%s): %s",
                           sid, removed["speaker"], removed.get("text", "")[:60])

        # Ensure last turn is agent (close the conversation)
        if turns and turns[-1].get("speaker") != "agent":
            # Find last agent turn and move it to end, or just swap last turn
            last_human_idx = len(turns) - 1
            # Try to find last agent turn before the human turn
            for i in range(len(turns) - 2, -1, -1):
                if turns[i].get("speaker") == "agent":
                    break
            else:
                # No agent turn found, change last to agent
                turns[-1]["speaker"] = "agent"
            logger.warning("Dialogue %s: fixed last turn to be agent", sid)

        # Renumber turns after any removals
        for i, t in enumerate(turns):
            t["turn"] = i + 1

        all_text = " ".join(t.get("text", "") for t in turns).lower()
        issues = []
        if agent_name.lower() not in all_text:
            issues.append(f"agent name '{agent_name}' not found in dialogue text")
        if human_name.lower() not in all_text:
            issues.append(f"human name '{human_name}' not found in dialogue text")
        if turns[0].get("speaker") != "agent":
            issues.append("turn 1 is not agent (must open)")
        if issues:
            logger.warning("Dialogue %s validation issues: %s", sid, "; ".join(issues))

    # Reject repetitive dialogues (LLM infinite loop + phrase repetition)
    if len(turns) >= 8:
        agent_texts = [t.get("text", "") for t in turns if t.get("speaker") == "agent"]
        human_texts = [t.get("text", "") for t in turns if t.get("speaker") == "human"]
        dup_count = 0
        for texts in [agent_texts, human_texts]:
            for i in range(1, len(texts)):
                if texts[i] and texts[i][:80] == texts[i - 1][:80]:
                    dup_count += 1
        if dup_count >= 5:
            logger.error("Dialogue %s rejected: %d repetitive turns", sid, dup_count)
            return None

    # Fix phrase-level repetition via LLM rewrite instead of rejecting
    fixed = await _fix_boring_turns(turns, router, sid)
    if fixed is None:
        logger.warning("Dialogue %s: boring phrases persist after fix attempts — rejecting", sid)
        return None
    turns = fixed

    dialogue["dialogue_id"] = dialogue_id
    dialogue["situation_id"] = sid
    dialogue["llm_backend"] = llm_backend

    # Quality scoring (lightweight LLM-as-judge)
    scores = await score_dialogue(dialogue, situation, router)
    if scores:
        dialogue["quality_scores"] = scores
        overall = scores.get("overall", 0)
        if overall < 3:
            logger.warning("Low quality dialogue %s: overall=%.1f — %s",
                           sid, overall, scores.get("reason", ""))

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
        return manifest_entry, (speaker, apply_fade(trim_leading_silence(audio)))

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
            "context_details": situation.get("context_details", {}),
            "characters": situation.get("characters", {}),
            "expected_emotion_arc": situation.get("expected_emotion_arc", []),
            "difficulty": situation.get("difficulty", ""),
            "key_details": situation.get("key_details", {}),
        },
        "quality_scores": dialogue.get("quality_scores"),
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

    def validate_situation(sit):
        """Validate + fix situation: swap agent/human if roles are reversed, fix TTS names."""
        if isinstance(sit, str):
            try:
                sit = json.loads(sit)
            except (json.JSONDecodeError, ValueError):
                return sit
        if not isinstance(sit, dict):
            return sit
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
            if "agent" in chars and "human" in chars:
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

    # Enrich TTS_VOICE_NAMES with all voice names from config
    for _pool_voices in voice_pools.values():
        for _vn in _pool_voices:
            TTS_VOICE_NAMES_UNDER.add(_vn)
            TTS_VOICE_NAMES_UNDER.add(_vn.replace(" ", "_"))
    TTS_VOICE_NAMES.update(TTS_VOICE_NAMES_UNDER | {n.replace("_", " ") for n in TTS_VOICE_NAMES_UNDER})

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
    n_llm_workers = min(max_concurrent, 8)
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
                        if "created_at" not in sit:
                            sit["created_at"] = "legacy"
                        total_situations += 1
                        batch_count += 1
                        await situation_queue.put((sit, run_suffix))
                        if max_dialogues > 0 and total_situations >= max_dialogues:
                            return batch_count
                except Exception as e:
                    logger.warning("Failed to load %s: %s", sf.name, e)

        if batch_count > 0:
            logger.info("Loaded %d existing situations", batch_count)

        # Generate new situations per theme — dispatch in parallel, feed queue incrementally
        themes_to_gen = []
        for theme in themes:
            if max_dialogues > 0 and total_situations >= max_dialogues:
                break
            theme_name = theme["theme"]
            theme_file = sit_dir / f"{theme_name}.json"
            if not theme_file.exists():
                themes_to_gen.append(theme)

        async def gen_one_theme(theme: dict) -> tuple[str, list[dict] | None, str | None, str]:
            theme_name = theme["theme"]
            n = min(pipeline_cfg.get("situations_per_theme", 20), 2)
            seed = random.randint(10000, 99999)
            prompt = SITUATION_PROMPT.format(
                n=n,
                seed=seed,
                theme_abbr=theme_name[:2],
                theme_json=json.dumps(theme, indent=2, ensure_ascii=False),
                voices_json=voices_json,
            )
            try:
                raw, backend_name = await router.generate(prompt)
                situations = parse_json_response(raw)
                return theme_name, situations, None, backend_name
            except Exception as e:
                logger.error("Situation generation failed for '%s': %s", theme_name, e)
                return theme_name, None, str(e), "unknown"

        if themes_to_gen:
            # Fire all tasks, feed queue as each completes — LLM workers start immediately
            pending = {asyncio.create_task(gen_one_theme(t)): t for t in themes_to_gen}
            while pending:
                done, pending_set = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                pending = {k: v for k, v in pending.items() if k not in done}
                for task in done:
                    theme_name, situations, error, sit_backend = task.result()
                    if error is not None:
                        fail_file = sit_dir / f"_failed_{theme_name}.json"
                        fail_entry = {"theme": theme_name, "error": error,
                                      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
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
                        # Filter out malformed situations (e.g. lists from LLM)
                        situations = [validate_situation(s) for s in situations if isinstance(s, dict)]
                        if not situations:
                            continue
                        logger.info("Generated %d situations for '%s' [%s]", len(situations), theme_name, sit_backend)
                        for s in situations:
                            s["llm_backend"] = sit_backend
                            s["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        theme_file = sit_dir / f"{theme_name}.json"
                        with open(theme_file, "w") as f:
                            json.dump(situations, f, indent=2, ensure_ascii=False)
                        for sit in situations:
                            total_situations += 1
                            batch_count += 1
                            await situation_queue.put((sit, run_suffix))
                            if max_dialogues > 0 and total_situations >= max_dialogues:
                                break


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
