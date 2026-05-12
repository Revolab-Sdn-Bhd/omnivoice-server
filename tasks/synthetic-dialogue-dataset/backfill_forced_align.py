#!/usr/bin/env python3
"""
Backfill words_forced into existing Stage 5 validation files.

For each turn in each validation JSON, if words_forced is missing or empty,
runs WhisperX forced alignment against the original spoken text and saves
the result. Turns that already have words_forced populated are skipped.

Usage:
    # Dry run - show what needs backfilling
    python backfill_forced_align.py --dry-run

    # Run backfill on all validation files
    python backfill_forced_align.py

    # Specific dialogue IDs only
    python backfill_forced_align.py --ids dlg_001 dlg_002

    # Custom alignment model
    python backfill_forced_align.py --align-model mesolitica/wav2vec2-xls-r-300m-mixed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.json") as f:
        return json.load(f)


def needs_backfill(turn_data: dict) -> bool:
    """Check if a turn needs words_forced backfilled."""
    forced = turn_data.get("words_forced", [])
    return len(forced) == 0


def load_align_session(asr_config: dict, align_model_override: str | None = None):
    import whisperx

    device = asr_config.get("device", "cuda")
    language = asr_config.get("language", "ms")

    align_model_name = align_model_override or asr_config.get("align_model")
    align_model = None
    align_metadata = None
    if align_model_name:
        logger.info("Loading alignment model %s...", align_model_name)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=device, model_name=align_model_name,
        )
    else:
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language, device=device,
            )
        except Exception:
            logger.warning("No default alignment model for language: %s", language)

    return {
        "whisperx": whisperx,
        "align_model": align_model,
        "align_metadata": align_metadata,
        "device": device,
    }


def run_forced_alignment(session: dict, audio_path: Path, original_text: str) -> list[dict]:
    """Run forced alignment of original text against audio, return word list."""
    whisperx = session["whisperx"]
    align_model = session["align_model"]
    align_metadata = session["align_metadata"]
    device = session["device"]

    if align_model is None:
        return []

    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000.0
    forced_input = [{"text": original_text, "start": 0.0, "end": duration}]

    result_forced = whisperx.align(
        forced_input, align_model, align_metadata, audio, device,
    )

    words = []
    for seg in result_forced.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", 0.0), 3),
                "end": round(w.get("end", 0.0), 3),
                "score": round(w.get("score", 0.0), 4),
            })
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill words_forced into validation files")
    parser.add_argument("--dry-run", action="store_true", help="Show what needs backfilling")
    parser.add_argument("--ids", nargs="+", help="Specific dialogue IDs to process")
    parser.add_argument("--align-model", type=str, default=None, help="Alignment model override")
    args = parser.parse_args()

    config = load_config()
    asr_config = config.get("asr", {})
    output_dir = Path(config["output_dir"])
    audio_dir = output_dir / "stage4_audio"
    validation_dir = output_dir / "stage5_validation"

    if not validation_dir.exists():
        logger.error("No stage5_validation directory found")
        sys.exit(1)

    # Scan all validation files
    total_files = 0
    total_turns_need = 0
    total_turns_ok = 0
    work_items = []

    for val_file in sorted(validation_dir.glob("*_validation.json")):
        with open(val_file) as f:
            validation = json.load(f)

        dialogue_id = validation.get("dialogue_id", val_file.stem.replace("_validation", ""))

        if args.ids and dialogue_id not in args.ids:
            continue

        turns = validation.get("turns", {})
        turns_needing = []

        for turn_key, turn_data in turns.items():
            if turn_data.get("status") not in ("pass", "fail"):
                continue
            if needs_backfill(turn_data):
                turns_needing.append(turn_key)
                total_turns_need += 1
            else:
                total_turns_ok += 1

        if turns_needing:
            total_files += 1
            work_items.append((val_file, dialogue_id, turns_needing))

    logger.info(
        "Scan: %d files need backfill (%d turns missing words_forced), %d turns already have it",
        total_files, total_turns_need, total_turns_ok,
    )

    if args.dry_run:
        for val_file, dialogue_id, turns in work_items[:20]:
            print(f"  {dialogue_id}: {len(turns)} turns need words_forced")
        if len(work_items) > 20:
            print(f"  ... and {len(work_items) - 20} more files")
        return

    if not work_items:
        logger.info("Nothing to backfill")
        return

    # Load WhisperX session
    session = load_align_session(asr_config, align_model_override=args.align_model)

    if session["align_model"] is None:
        logger.error("No alignment model available, cannot backfill")
        sys.exit(1)

    # Process each file
    processed_files = 0
    processed_turns = 0
    failed_turns = 0

    for val_file, dialogue_id, turns_needing in work_items:
        dialogue_dir = audio_dir / dialogue_id

        with open(val_file) as f:
            validation = json.load(f)

        # Load manifest to get turn metadata
        manifest_path = dialogue_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("No manifest for %s, skipping", dialogue_id)
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Build turn_key -> (wav_path, spoken_text) lookup
        turn_lookup = {}
        for mt in manifest["turns"]:
            tk = f"turn_{mt['turn']:02d}_{mt['speaker']}"
            wav_name = mt.get("wav_file", f"turn_{mt['turn']:02d}_{mt['speaker']}.wav")
            wav_path = dialogue_dir / wav_name
            spoken = mt.get("text_spoken", mt["text_written"])
            turn_lookup[tk] = (wav_path, spoken)

        changed = False
        for turn_key in turns_needing:
            if turn_key not in turn_lookup:
                continue
            if turn_key not in validation["turns"]:
                continue

            wav_path, spoken_text = turn_lookup[turn_key]

            if not wav_path.exists():
                logger.warning("Missing WAV %s for %s/%s", wav_path.name, dialogue_id, turn_key)
                failed_turns += 1
                continue

            try:
                words_forced = run_forced_alignment(session, wav_path, spoken_text)
                if words_forced:
                    validation["turns"][turn_key]["words_forced"] = words_forced
                    changed = True
                    processed_turns += 1
                else:
                    asr_words = validation["turns"][turn_key].get("words", [])
                    if asr_words:
                        validation["turns"][turn_key]["words_forced"] = asr_words
                        changed = True
                        processed_turns += 1
                        logger.info("Fallback to ASR words for %s/%s (%d words)", dialogue_id, turn_key, len(asr_words))
                    else:
                        failed_turns += 1
            except Exception as e:
                logger.exception("Failed %s/%s: %s", dialogue_id, turn_key, e)
                failed_turns += 1

        if changed:
            with open(val_file, "w") as f:
                json.dump(validation, f, indent=2, ensure_ascii=False)
            processed_files += 1

        if processed_files % 50 == 0:
            logger.info("Progress: %d/%d files, %d turns backfilled, %d failed",
                        processed_files, len(work_items), processed_turns, failed_turns)

    logger.info(
        "Backfill complete: %d files updated, %d turns backfilled, %d failed",
        processed_files, processed_turns, failed_turns,
    )


if __name__ == "__main__":
    main()
