#!/usr/bin/env python3
"""
Upload M-CHAT final dataset to HuggingFace (Revolab/omnidialog).

Uses huggingface_hub Python API for batch uploads.
data_stereo/ is sharded into subdirectories by theme prefix (first 2 chars)
to stay under HF's 10,000 files-per-directory limit.

Usage:
    # Full upload (creates repo, uploads everything)
    python upload_to_hf.py

    # Incremental upload (only uploads files not yet in the repo)
    python upload_to_hf.py --incremental

    # Dry run
    python upload_to_hf.py --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)
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
omnidialog.jsonl              — {path, duration, llm_backend, theme} per dialogue
data_stereo/
  {shard}/                    — sharded by theme prefix (2-char) to stay under 10K files/dir
    {id}.mp3                  — stereo audio
    {id}.json                 — transcript + word timestamps + situation metadata
stats.json                    — aggregate statistics
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
| path | string | Relative path to MP3 in data_stereo/{shard}/ |
| duration | float | Duration in seconds |
| llm_backend | string | LLM used (zai/qwen3) |
| theme | string | Dialogue theme category |

### data_stereo/{shard}/{id}.json

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


def get_shard_key(filename: str) -> str:
    return filename[:2]


def build_staging_dir(final_dir: Path, skip_existing: set[str] | None = None) -> Path:
    """Create a temp dir with sharded data_stereo structure using symlinks."""
    staging = Path(tempfile.mkdtemp(prefix="omnidialog_upload_"))

    # Sharded JSONL with merged word-level alignment
    src = final_dir / "dataset.jsonl"
    data_dir = final_dir / "data_stereo"
    if src.exists():
        lines = src.read_text().strip().split("\n")
        sharded = []
        merged = 0
        for line in lines:
            rec = json.loads(line)
            old_path = rec.get("path", "")
            if old_path.startswith("data_stereo/"):
                fname = old_path[len("data_stereo/"):]
                shard = get_shard_key(fname)
                rec["path"] = f"data_stereo/{shard}/{fname}"
                # Merge companion JSON with word-level alignment
                base = fname.rsplit(".", 1)[0]
                companion = data_dir / f"{base}.json"
                if companion.exists():
                    try:
                        with open(companion) as cf:
                            comp = json.load(cf)
                        rec["turns"] = comp.get("turns", [])
                        rec["situation"] = comp.get("situation", {})
                        rec["dialogue_id"] = comp.get("dialogue_id", base)
                        merged += 1
                    except Exception as e:
                        logger.warning("Failed to merge companion %s: %s", companion.name, e)
            sharded.append(json.dumps(rec, ensure_ascii=False))
        (staging / "omnidialog.jsonl").write_text("\n".join(sharded) + "\n")
        logger.info("Built sharded JSONL (%d entries, %d with word alignment)", len(sharded), merged)

    # README
    (staging / "README.md").write_text(DATASET_CARD)

    # Stats
    stats_src = final_dir / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, staging / "stats.json")

    # data_stereo sharded via symlinks
    data_dir = final_dir / "data_stereo"
    if data_dir.exists():
        all_files = sorted(f for f in os.listdir(data_dir) if not f.startswith("."))
        n_linked = 0
        for fname in all_files:
            shard = get_shard_key(fname)
            repo_path = f"data_stereo/{shard}/{fname}"
            if skip_existing and repo_path in skip_existing:
                continue
            shard_dir = staging / "data_stereo" / shard
            shard_dir.mkdir(parents=True, exist_ok=True)
            src_path = data_dir / fname
            os.symlink(src_path, shard_dir / fname)
            n_linked += 1
        logger.info("Linked %d / %d data files into staging dir", n_linked, len(all_files))

    return staging


def upload(args: argparse.Namespace) -> None:
    final_dir = FINAL_DIR
    if not final_dir.exists():
        logger.error("Final directory not found: %s", final_dir)
        raise SystemExit(1)

    repo_id = args.repo

    from huggingface_hub import HfApi
    api = HfApi()

    if args.dry_run:
        data_dir = final_dir / "data_stereo"
        n_data = len([f for f in os.listdir(data_dir) if not f.startswith(".")]) if data_dir.exists() else 0
        logger.info("Would upload to %s (private):", repo_id)
        logger.info("  README.md")
        logger.info("  omnidialog.jsonl (sharded paths, %d entries)", n_data // 2)
        logger.info("  stats.json")
        logger.info("  data_stereo/ (%d files, sharded by prefix)", n_data)
        return

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    skip = None
    if args.incremental:
        existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
        logger.info("Repo has %d existing files", len(existing))
        skip = existing

    staging = build_staging_dir(final_dir, skip_existing=skip)

    try:
        # Upload index files separately first (small, fast)
        for f in ["README.md", "omnidialog.jsonl", "stats.json"]:
            src = staging / f
            if src.exists():
                logger.info("Uploading %s...", f)
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=f,
                    repo_id=repo_id,
                    repo_type="dataset",
                )

        # Upload sharded data_stereo
        ds_staging = staging / "data_stereo"
        if ds_staging.exists():
            shard_dirs = sorted(d for d in ds_staging.iterdir() if d.is_dir())
            for shard_dir in shard_dirs:
                n = len(list(shard_dir.iterdir()))
                if n == 0:
                    continue
                logger.info("Uploading data_stereo/%s/ (%d files)...", shard_dir.name, n)
                api.upload_folder(
                    folder_path=str(shard_dir),
                    path_in_repo=f"data_stereo/{shard_dir.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            logger.info("Uploaded %d shards.", len(shard_dirs))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

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
