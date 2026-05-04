#!/usr/bin/env python3
"""
Generate voice design audio samples using OmniVoice instruct mode.

Outputs WAV + .txt pairs into static/speakers/ so they can be used
as voice clone references later.

Usage:
    uv run python scripts/generate_voice_design.py
    uv run python scripts/generate_voice_design.py --sample-text "Hello world" --instruct "Female, Young Adult, American Accent"
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
OUTPUT_DIR = ROOT / "omnivoice_server" / "static" / "speakers"

# Default voice designs to generate
DEFAULT_VOICES = [
    {
        "name": "female_young_us",
        "instruct": "Female, Young Adult, Moderate Pitch, American Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
    {
        "name": "male_young_us",
        "instruct": "Male, Young Adult, Moderate Pitch, American Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
    {
        "name": "female_middle_my",
        "instruct": "Female, Middle-aged, Moderate Pitch, Malay Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
    {
        "name": "male_elderly_uk",
        "instruct": "Male, Elderly, Low Pitch, British Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
    {
        "name": "female_child",
        "instruct": "Female, Child, High Pitch, American Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
    {
        "name": "male_deep_us",
        "instruct": "Male, Young Adult, Very Low Pitch, American Accent",
        "text": "Hello, welcome to SepBox text to speech service. How can I help you today?",
    },
]


def load_model(model_id: str, device: str) -> OmniVoice:
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    logger.info("Loading model %s on %s ...", model_id, device)
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    logger.info("Model loaded.")
    return model


def generate_one(
    model: OmniVoice,
    instruct: str,
    text: str,
    num_step: int = 16,
    class_temperature: float = 0.0,
) -> torch.Tensor:
    logger.info("Generating: instruct=%r  text=%r", instruct, text[:60])
    audio = model.generate(
        text=text,
        instruct=instruct,
        num_step=num_step,
        class_temperature=class_temperature,
    )
    # audio is list of tensors [(1, T), ...]
    return torch.cat([t.cpu() for t in audio], dim=-1)


def save_wav(tensor: torch.Tensor, path: Path) -> None:
    torchaudio.save(
        str(path),
        tensor.cpu(),
        SAMPLE_RATE,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )
    logger.info("Saved %s (%.1fs, %d KB)", path.name, tensor.shape[-1] / SAMPLE_RATE, path.stat().st_size // 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate voice design audio samples")
    parser.add_argument("--model", default="Revolab/omnivoice", help="Model checkpoint or HF repo")
    parser.add_argument("--device", default=None, help="Device (auto-detected if omitted)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory for WAV files")
    parser.add_argument("--num-step", type=int, default=16, help="Inference steps")
    parser.add_argument("--temperature", type=float, default=0.0, help="Class temperature")
    parser.add_argument("--name", help="Voice name (single generation mode)")
    parser.add_argument("--instruct", help="Instruct string (single generation mode)")
    parser.add_argument("--sample-text", help="Text to synthesize (single generation mode)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, device)

    # Single generation mode
    if args.instruct:
        name = args.name or "custom_design"
        text = args.sample_text or "Hello, this is a voice design sample."
        tensor = generate_one(model, args.instruct, text, args.num_step, args.temperature)
        wav_path = out_dir / f"{name}.wav"
        txt_path = out_dir / f"{name}.txt"
        save_wav(tensor, wav_path)
        txt_path.write_text(text.strip() + "\n")
        logger.info("Saved ref_text to %s", txt_path.name)
        return

    # Batch mode — generate all default voices
    logger.info("Generating %d voice design samples ...", len(DEFAULT_VOICES))
    for voice in DEFAULT_VOICES:
        wav_path = out_dir / f"{voice['name']}.wav"
        txt_path = out_dir / f"{voice['name']}.txt"

        if wav_path.exists():
            logger.info("Skipping %s (already exists)", voice["name"])
            continue

        tensor = generate_one(
            model, voice["instruct"], voice["text"],
            args.num_step, args.temperature,
        )
        save_wav(tensor, wav_path)
        txt_path.write_text(voice["text"].strip() + "\n")
        logger.info("Saved ref_text to %s", txt_path.name)

    logger.info("Done. %d voices in %s", len(DEFAULT_VOICES), out_dir)


if __name__ == "__main__":
    main()
