#!/usr/bin/env python3
"""
Generate all voice design combinations using OmniVoice instruct mode.

Loads model once, then generates every combination of:
  Gender × Age × Pitch × Accent × Style

Outputs WAV + .txt pairs into designed_voice/ for audition.

Usage:
    uv run python scripts/generate_all_designs.py
    uv run python scripts/generate_all_designs.py --skip-existing
"""

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
OUTPUT_DIR = ROOT / "designed_voice"

GENDERS = ["Male", "Female"]
AGES = ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"]
PITCHES = ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"]
ACCENTS = [
    "American Accent",
    "British Accent",
    "Australian Accent",
    "Chinese Accent",
    "Canadian Accent",
    "Indian Accent",
    "Korean Accent",
    "Portuguese Accent",
    "Russian Accent",
    "Japanese Accent",
]
STYLES = ["", "Whisper"]

SAMPLE_TEXT = "Hello, welcome to SepBox text to speech service. How can I help you today?"


def load_model(model_id: str, device: str) -> OmniVoice:
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    logger.info("Loading model %s on %s ...", model_id, device)
    model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype, load_asr=False)
    logger.info("Model loaded.")
    return model


def generate_one(model, instruct, text, num_step=16):
    audio = model.generate(text=text, instruct=instruct, num_step=num_step)
    return torch.cat([t.cpu() for t in audio], dim=-1)


def save_wav(tensor, path):
    torchaudio.save(
        str(path), tensor.cpu(), SAMPLE_RATE,
        format="wav", encoding="PCM_S", bits_per_sample=16,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate all voice design combinations")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--text", default=SAMPLE_TEXT)
    parser.add_argument(
        "--scope",
        choices=["all", "gender_age", "gender_accent", "pitch"],
        default="all",
        help="all=gender×age×accent, gender_age=gender×age only, gender_accent=gender×accent only, pitch=pitch sweep",
    )
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build combinations based on scope
    combos = []
    if args.scope == "all":
        # Gender × Age × Accent (moderate pitch, no style)
        for g, a, acc in itertools.product(GENDERS, AGES, ACCENTS):
            name = f"{g.lower()}_{a.lower().replace('-','')}_moderate_{acc.lower().replace(' ','_')}"
            instruct = f"{g}, {a}, Moderate Pitch, {acc}"
            combos.append((name, instruct))
        # Gender × Pitch (young adult, american, no style)
        for g, p in itertools.product(GENDERS, PITCHES):
            name = f"{g.lower()}_young_adult_{p.lower().replace(' ','_')}_american"
            instruct = f"{g}, Young Adult, {p}, American Accent"
            combos.append((name, instruct))
        # Style variations (young adult, moderate, american)
        for g in GENDERS:
            name = f"{g.lower()}_young_adult_moderate_american_whisper"
            instruct = f"{g}, Young Adult, Moderate Pitch, American Accent, Whisper"
            combos.append((name, instruct))

    elif args.scope == "gender_age":
        for g, a in itertools.product(GENDERS, AGES):
            name = f"{g.lower()}_{a.lower().replace('-','')}"
            instruct = f"{g}, {a}, Moderate Pitch, American Accent"
            combos.append((name, instruct))

    elif args.scope == "gender_accent":
        for g, acc in itertools.product(GENDERS, ACCENTS):
            name = f"{g.lower()}_{acc.lower().replace(' ','_')}"
            instruct = f"{g}, Young Adult, Moderate Pitch, {acc}"
            combos.append((name, instruct))

    elif args.scope == "pitch":
        for g, p in itertools.product(GENDERS, PITCHES):
            name = f"{g.lower()}_{p.lower().replace(' ','_')}"
            instruct = f"{g}, Young Adult, {p}, American Accent"
            combos.append((name, instruct))

    logger.info("Total combinations: %d", len(combos))

    model = load_model(args.model, device)
    text = args.text
    generated = 0
    skipped = 0
    start = time.time()

    for i, (name, instruct) in enumerate(combos):
        wav_path = out_dir / f"{name}.wav"
        txt_path = out_dir / f"{name}.txt"

        if args.skip_existing and wav_path.exists():
            skipped += 1
            continue

        try:
            tensor = generate_one(model, instruct, text, args.num_step)
            save_wav(tensor, wav_path)
            txt_path.write_text(text.strip() + "\n")
            generated += 1
            dur = tensor.shape[-1] / SAMPLE_RATE
            logger.info(
                "[%d/%d] %s (%.1fs) ← %s",
                i + 1, len(combos), name, dur, instruct,
            )
        except Exception as e:
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(combos), name, e)

    elapsed = time.time() - start
    logger.info(
        "Done. %d generated, %d skipped in %.0fs → %s",
        generated, skipped, elapsed, out_dir,
    )


if __name__ == "__main__":
    main()
