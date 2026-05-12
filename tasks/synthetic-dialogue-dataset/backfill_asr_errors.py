#!/usr/bin/env python3
"""Backfill words_forced for turns with asr_error status."""

import json
import logging
from pathlib import Path

import whisperx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/mnt/data/work/omnivoice-server/output/synthetic-dialogue")


def load_config():
    with open(SCRIPT_DIR / "config.json") as f:
        return json.load(f)


def run_forced_alignment(align_model, align_metadata, audio_path, text, device):
    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000.0
    forced_input = [{"text": text, "start": 0.0, "end": duration}]
    result = whisperx.align(forced_input, align_model, align_metadata, audio, device)
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", 0.0), 3),
                "end": round(w.get("end", 0.0), 3),
                "score": round(w.get("score", 0.0), 4),
            })
    return words


def main():
    config = load_config()
    asr_config = config.get("asr", {})
    audio_dir = OUTPUT_DIR / "stage4_audio"
    val_dir = OUTPUT_DIR / "stage5_validation"

    targets = []
    for val_file in sorted(val_dir.glob("*_validation.json")):
        with open(val_file) as f:
            val = json.load(f)
        dlg_id = val.get("dialogue_id", val_file.stem.replace("_validation", ""))
        for turn_key, td in val.get("turns", {}).items():
            if td.get("status") not in ("asr_error", "fail"):
                continue
            if td.get("words_forced"):
                continue
            targets.append((val_file, dlg_id, turn_key))

    logger.info("Found %d turns needing forced alignment", len(targets))
    if not targets:
        return

    device = asr_config.get("device", "cuda")
    language = asr_config.get("language", "ms")
    align_model_name = asr_config.get("align_model")
    logger.info("Loading alignment model %s...", align_model_name)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device, model_name=align_model_name,
    )

    by_dlg = {}
    for val_file, dlg_id, turn_key in targets:
        by_dlg.setdefault(dlg_id, []).append((val_file, turn_key))

    processed = 0
    failed = 0

    for dlg_id, items in by_dlg.items():
        dlg_dir = audio_dir / dlg_id
        manifest_path = dlg_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("No manifest for %s", dlg_id)
            failed += len(items)
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        turn_lookup = {}
        for mt in manifest["turns"]:
            tk = f"turn_{mt['turn']:02d}_{mt['speaker']}"
            wav_name = mt.get("wav_file", f"turn_{mt['turn']:02d}_{mt['speaker']}.wav")
            turn_lookup[tk] = (dlg_dir / wav_name, mt.get("text_spoken", mt["text_written"]))

        changed_files = {}
        for val_file, turn_key in items:
            wav_path, text_spoken = turn_lookup.get(turn_key, (None, None))
            if not wav_path or not wav_path.exists():
                logger.warning("Missing WAV for %s/%s", dlg_id, turn_key)
                failed += 1
                continue

            try:
                words = run_forced_alignment(align_model, align_metadata, wav_path, text_spoken, device)
                if words:
                    if val_file not in changed_files:
                        with open(val_file) as f:
                            changed_files[val_file] = json.load(f)
                    changed_files[val_file]["turns"][turn_key]["words_forced"] = words
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.exception("Failed %s/%s: %s", dlg_id, turn_key, e)
                failed += 1

        for val_file, val_data in changed_files.items():
            with open(val_file, "w") as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)

        if processed % 100 == 0 and processed > 0:
            logger.info("Progress: %d done, %d failed", processed, failed)

    logger.info("Complete: %d turns aligned, %d failed", processed, failed)


if __name__ == "__main__":
    main()
