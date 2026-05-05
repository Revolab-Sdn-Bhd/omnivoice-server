#!/usr/bin/env python3
"""
Upload M-CHAT final dataset to HuggingFace (Revolab/omnidialog).

Uses `hf upload` CLI for simplicity. The HF repo is PRIVATE.

Usage:
    # Full upload
    python upload_to_hf.py

    # Incremental upload (re-uploads index files, only new data_stereo files)
    python upload_to_hf.py --incremental

    # Dry run
    python upload_to_hf.py --dry-run
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
FINAL_DIR = PROJECT_DIR / "output" / "synthetic-dialogue" / "final"

DATASET_CARD = """\
---
license: cc-by-nc-4.0
task_categories:
  - automatic-speech-recognition
  - text-to-speech
language:
  - ms
  - en
tags:
  - spoken-dialogue
  - code-mixing
  - bahasa-melayu
  - synthetic
  - multi-speaker
  - stereo
size_categories:
  - 1K<n<10K
---

# OmniDialog — Malay Conversational Dialogue Dataset

Synthetic multi-turn spoken dialogue dataset for Bahasa Melayu + English code-mixing,
generated via LLM → TTS pipeline with ASR validation.

## Format

- **Stereo MP3 (192kbps)**: Left channel = agent, Right channel = human
- **Word-level timestamps** via WhisperX alignment
- **Written + spoken text** per turn (revo-norm normalized)

## Structure

```
omnidialog.jsonl          — {path, duration, llm_backend, theme} per dialogue
data_stereo/
  {id}.mp3                — stereo audio
  {id}.json               — transcript + word timestamps + situation metadata
stats.json                — aggregate statistics
```

## Generation Pipeline

```
Theme → Situation → Dialogue (LLM) → TTS (OmnIvoice) → ASR Validation (WhisperX)
```

- **LLM backends**: zai (glm-4.5-air), qwen3 (Qwen3-30B-A3B Malaysian)
- **TTS**: OmnIvoice with voice cloning (6 speakers, gender-pooled)
- **ASR**: WhisperX large-v3 with word-level alignment
- **WER threshold**: <0.50, **CER threshold**: <0.40

## Fields

### omnidialog.jsonl

| Field | Type | Description |
|-------|------|-------------|
| path | string | Relative path to MP3 in data_stereo/ |
| duration | float | Duration in seconds |
| llm_backend | string | LLM used (zai/qwen3) |
| theme | string | Dialogue theme category |

### data_stereo/{id}.json

| Field | Type | Description |
|-------|------|-------------|
| dialogue_id | string | Unique identifier |
| llm_backend | string | LLM backend used |
| situation | object | Scenario, characters, key details |
| turns | array | Per-turn transcript with word timestamps |

Each turn:
| Field | Type | Description |
|-------|------|-------------|
| turn | int | Turn number (1-indexed) |
| speaker | string | "agent" or "human" |
| channel | string | "left" or "right" |
| text_written | string | Original LLM output |
| text_spoken | string | Normalized spoken form (TTS input) |
| start_s | float | Start time in stereo audio |
| end_s | float | End time in stereo audio |
| words | array | Word-level {word, start, end, score} |

## Themes

customer_support, food_dining, transport, retail, government_services, workplace,
real_estate, business, finance, family, neighbourhood, religious, sports_leisure,
medical, mental_health, wellness, education, skills_training, podcast, radio_show,
content_creation, emergency, legal, debate_discussion, hotel_hospitality, insurance,
auto_mechanic, immigration, property_agent, tuition_center, wedding_planning, fitness_gym

## License

CC BY-NC 4.0 — for research and non-commercial use.
"""


def hf_upload(repo_id: str, local_path: str, path_in_repo: str, dry_run: bool = False) -> None:
    cmd = [
        "hf", "upload", repo_id, local_path,
        path_in_repo, "--repo-type", "dataset",
    ]
    if dry_run:
        logger.info("DRY RUN: %s", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def upload(args: argparse.Namespace) -> None:
    final_dir = FINAL_DIR
    if not final_dir.exists():
        logger.error("Final directory not found: %s", final_dir)
        sys.exit(1)

    repo_id = args.repo

    # Write README to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(DATASET_CARD)
        readme_path = f.name

    if args.dry_run:
        data_dir = final_dir / "data_stereo"
        n_data = len(list(data_dir.iterdir())) if data_dir.exists() else 0
        logger.info("Would upload to %s (private):", repo_id)
        logger.info("  README.md")
        logger.info("  omnidialog.jsonl")
        logger.info("  stats.json")
        logger.info("  data_stereo/ (%d files)", n_data)
        Path(readme_path).unlink()
        return

    # README
    logger.info("Uploading README.md...")
    hf_upload(repo_id, readme_path, "README.md")
    Path(readme_path).unlink()

    # Index files (always re-upload — they change every cycle)
    logger.info("Uploading omnidialog.jsonl...")
    hf_upload(repo_id, str(final_dir / "dataset.jsonl"), "omnidialog.jsonl")

    logger.info("Uploading stats.json...")
    hf_upload(repo_id, str(final_dir / "stats.json"), "stats.json")

    # data_stereo folder (CLI skips unchanged files automatically)
    data_dir = final_dir / "data_stereo"
    if data_dir.exists():
        n_files = len(list(data_dir.iterdir()))
        logger.info("Uploading data_stereo/ (%d files)...", n_files)
        cmd = [
            "hf", "upload", repo_id, str(data_dir),
            "data_stereo", "--repo-type", "dataset",
        ]
        subprocess.run(cmd, check=True)

    logger.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload M-CHAT dataset to HuggingFace")
    parser.add_argument("--repo", default="Revolab/omnidialog", help="HF dataset repo ID")
    parser.add_argument("--incremental", action="store_true", help="Incremental upload")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()
    upload(args)


if __name__ == "__main__":
    main()
