#!/usr/bin/env python3
"""
Backfill situation generation via qwen3 (local vLLM).

Re-generates situations that were created by zai, so we get qwen3 coverage.
Old situations without llm_backend field are treated as "zai" (legacy).

Usage:
    # Dry run - show what would be backfilled
    python backfill_context.py --dry-run

    # Backfill all themes via qwen3
    python backfill_context.py

    # Backfill specific themes only
    python backfill_context.py --themes transport,food_dining

    # Force regenerate even qwen3 situations
    python backfill_context.py --force
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.json") as f:
        return json.load(f)


async def generate_situation_via_qwen3(theme: dict, config: dict) -> list[dict] | None:
    """Generate situations using local vLLM (qwen3)."""
    qwen3_cfg = None
    for b in config.get("llm_backends", []):
        if b.get("type") == "openai":
            qwen3_cfg = b
            break
    if not qwen3_cfg:
        logger.error("No openai/vLLM backend found in config")
        return None

    pipeline_cfg = config.get("pipeline", {})
    n = pipeline_cfg.get("situations_per_theme", 5)
    seed = random.randint(10000, 99999)
    theme_name = theme["theme"]

    voices_json = json.dumps(config["tts"]["voice_pools"])

    prompt = f"""You are generating training data for a Malay spoken dialogue system (M-CHAT).

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
- situation_id: "{theme_name[:2]}_{{domain}}_NNN" format
- theme, domain, scenario, context
- context_details: {{ "location": "...", "environment": "...",
  "circumstances": "...", "time_of_day": "..." }}
  - location: where it takes place
  - environment: channel/medium (e.g., "phone call", "video call", "chat app")
    IMPORTANT: All conversations are remote. Never use "face-to-face".
  - circumstances: what led to this conversation
  - time_of_day: e.g., "morning rush hour", "afternoon", "late night"
- characters: {{ "agent": {{...}}, "human": {{...}} }} with:
  name, gender, role, speaking_style, personality_traits, background,
  emotional_state, goals, communication_quirks
- expected_emotion_arc, difficulty, key_details

Theme config:
{json.dumps(theme, indent=2, ensure_ascii=False)}

Output ONLY the JSON array."""

    async with httpx.AsyncClient(
        base_url=qwen3_cfg.get("api_base", "http://localhost:8900/v1"),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": qwen3_cfg["model"],
                        "max_tokens": qwen3_cfg.get("max_tokens", 4096),
                        "messages": [{"role": "user", "content": prompt}],
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    },
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"].strip()
                import re
                text = re.sub(r"<think[\s\S]*?</think\s*>", "", text).strip()
                situations = json.loads(text)
                if isinstance(situations, dict):
                    situations = situations.get("situations", [situations])
                for s in situations:
                    s["llm_backend"] = "qwen3"
                return situations
            except Exception as e:
                logger.warning("Attempt %d for '%s': %s", attempt + 1, theme_name, e)
                await asyncio.sleep(3 * (attempt + 1))
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill situations via qwen3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backfilled")
    parser.add_argument("--themes", type=str, help="Comma-separated theme names")
    parser.add_argument("--force", action="store_true", help="Regenerate even qwen3 situations")
    args = parser.parse_args()

    config = load_config()
    output_dir = Path(config["output_dir"])
    sit_dir = output_dir / "stage2_situations"

    # Load themes
    with open(output_dir / "stage1_themes.json") as f:
        themes = json.load(f)

    if args.themes:
        theme_names = set(args.themes.split(","))
        themes = [t for t in themes if t["theme"] in theme_names]

    to_backfill = []
    for theme in themes:
        theme_name = theme["theme"]
        theme_file = sit_dir / f"{theme_name}.json"

        if not theme_file.exists():
            to_backfill.append((theme, "no_file"))
            continue

        with open(theme_file) as f:
            situations = json.load(f)

        backends = set(s.get("llm_backend", "zai") for s in situations)
        if args.force or backends == {"zai"} or backends == {"unknown"}:
            to_backfill.append((theme, f"backends={backends}"))

    logger.info("Themes to backfill: %d / %d", len(to_backfill), len(themes))

    if args.dry_run:
        for theme, reason in to_backfill:
            print(f"  {theme['theme']} ({reason})")
        return

    # Run 4 themes concurrently to keep vLLM busy
    sem = asyncio.Semaphore(4)

    async def backfill_one(theme, reason):
        async with sem:
            theme_name = theme["theme"]
            logger.info("Backfilling '%s' (%s)...", theme_name, reason)
            situations = await generate_situation_via_qwen3(theme, config)
            if situations:
                theme_file = sit_dir / f"{theme_name}.json"
                with open(theme_file, "w") as f:
                    json.dump(situations, f, indent=2, ensure_ascii=False)
                logger.info("  -> %d situations saved [%s]", len(situations), "qwen3")
            else:
                logger.error("  -> FAILED for '%s'", theme_name)

    await asyncio.gather(*[backfill_one(t, r) for t, r in to_backfill])

    logger.info("Backfill complete")


if __name__ == "__main__":
    asyncio.run(main())
