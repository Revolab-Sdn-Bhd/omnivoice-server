#!/usr/bin/env python3
"""Regenerate bad voice design files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
from omnivoice import OmniVoice
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

BAD_FILES = [
    # (subfolder, filename_stem, instruct)
    ("american_accent", "female_child", "Female, Child, Moderate Pitch, American Accent"),
    ("american_accent", "female_teenager", "Female, Teenager, Moderate Pitch, American Accent"),
    ("american_accent", "male_child", "Male, Child, Moderate Pitch, American Accent"),
    ("american_accent", "male_teenager", "Male, Teenager, Moderate Pitch, American Accent"),
    ("australian_accent", "female_child", "Female, Child, Moderate Pitch, Australian Accent"),
    ("australian_accent", "male_child", "Male, Child, Moderate Pitch, Australian Accent"),
    ("british_accent", "male_child", "Male, Child, Moderate Pitch, British Accent"),
    ("canadian_accent", "female_child", "Female, Child, Moderate Pitch, Canadian Accent"),
    ("canadian_accent", "female_teenager", "Female, Teenager, Moderate Pitch, Canadian Accent"),
    ("canadian_accent", "male_child", "Male, Child, Moderate Pitch, Canadian Accent"),
    ("indian_accent", "female_child", "Female, Child, Moderate Pitch, Indian Accent"),
    ("indian_accent", "female_middleaged", "Female, Middle-aged, Moderate Pitch, Indian Accent"),
    ("indian_accent", "female_teenager", "Female, Teenager, Moderate Pitch, Indian Accent"),
    ("indian_accent", "female_youngadult", "Female, Young Adult, Moderate Pitch, Indian Accent"),
    ("indian_accent", "male_child", "Male, Child, Moderate Pitch, Indian Accent"),
    ("indian_accent", "male_teenager", "Male, Teenager, Moderate Pitch, Indian Accent"),
    ("pitch", "female_low_pitch", "Female, Young Adult, Low Pitch, American Accent"),
    ("pitch", "female_moderate_pitch", "Female, Young Adult, Moderate Pitch, American Accent"),
]

TEXT = "Hello, welcome to SepBox text to speech service. How can I help you today?"

# Find actual filenames (handle spaces in filenames)
design_dir = ROOT / "designed_voice"
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16, load_asr=False)
logger.info("Model loaded. Regenerating %d files ...", len(BAD_FILES))

for subfolder, stem, instruct in BAD_FILES:
    folder = design_dir / subfolder
    # Find matching wav (stem might have spaces)
    matches = list(folder.glob(f"{stem}*.wav"))
    if not matches:
        logger.warning("Not found: %s/%s*.wav", subfolder, stem)
        continue
    wav_path = matches[0]
    txt_path = wav_path.with_suffix(".txt")

    for attempt in range(3):
        audio = model.generate(text=TEXT, instruct=instruct, num_step=16)
        tensor = torch.cat([t.cpu() for t in audio], dim=-1)
        dur = tensor.shape[-1] / SAMPLE_RATE

        # Check if output is silence (very low energy)
        energy = tensor.abs().mean().item()
        if energy < 0.001:
            logger.warning("  Attempt %d: silence (energy=%.4f), retrying", attempt + 1, energy)
            continue

        torchaudio.save(str(wav_path), tensor, SAMPLE_RATE, format="wav", encoding="PCM_S", bits_per_sample=16)
        txt_path.write_text(TEXT + "\n")
        logger.info("  OK %s/%s (%.1fs, energy=%.3f) ← %s", subfolder, wav_path.stem, dur, energy, instruct)
        break
    else:
        logger.error("  FAILED after 3 attempts: %s/%s", subfolder, wav_path.stem)

logger.info("Done.")
