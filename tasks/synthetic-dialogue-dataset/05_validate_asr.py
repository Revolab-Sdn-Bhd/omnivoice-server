#!/usr/bin/env python3
"""
Stage 5 — ASR validation via WhisperX + final Moshi-compatible dataset assembly.

Per-turn validation: runs WhisperX on each individual turn WAV, compares
against written text, computes WER/CER. Failed dialogues are scrubbed
(manifest deleted) so Stage 4 can re-generate them.

Reads:  stage4_audio/{dialogue_id}/manifest.json
Writes: stage5_validation/{dialogue_id}_validation.json
        stage5_validation/summary.json
        stage5_validation/failed_ids.json
        final/dataset.jsonl
        final/data_stereo/{dialogue_id}.mp3 + {dialogue_id}.json
        final/stats.json

Usage:
    # Full validation + dataset assembly
    python 05_validate_asr.py

    # Only scrub failures (delete bad outputs for re-generation)
    python 05_validate_asr.py --scrub-only

    # Skip WhisperX, just assemble final dataset from existing manifests
    python 05_validate_asr.py --skip-asr
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 24_000


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.json") as f:
        return json.load(f)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using jiwer."""
    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        logger.warning("jiwer not installed. Install: pip install jiwer")
        return -1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate using jiwer."""
    try:
        import jiwer
        return jiwer.cer(reference, hypothesis)
    except ImportError:
        return -1.0


class WhisperXSession:
    """Loads WhisperX models once, reuses across all turns."""

    def __init__(self, asr_config: dict, align_model_override: str | None = None):
        import whisperx

        self.whisperx = whisperx
        self.device = asr_config.get("device", "cuda")
        self.language = asr_config.get("language", "ms")
        self.asr_language = asr_config.get("asr_language", None)
        model_name = asr_config.get("model", "large-v3")
        compute_type = asr_config.get("compute_type", "float16")

        logger.info("Loading WhisperX model %s...", model_name)
        self.model = whisperx.load_model(
            model_name, self.device, compute_type=compute_type, language=self.asr_language,
        )

        align_model_name = align_model_override or asr_config.get("align_model")
        self.align_model = None
        self.align_metadata = None
        if align_model_name:
            logger.info("Loading alignment model %s...", align_model_name)
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language, device=self.device, model_name=align_model_name,
            )
        else:
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language, device=self.device,
                )
            except Exception:
                logger.warning("No default alignment model for language: %s", self.language)

    def transcribe(self, audio_path: Path) -> dict | None:
        try:
            audio = self.whisperx.load_audio(str(audio_path))
            result = self.model.transcribe(audio, batch_size=16, language=self.asr_language)

            segments = result.get("segments", [])
            text = " ".join(seg["text"].strip() for seg in segments)

            words = []
            if self.align_model is not None:
                try:
                    result_aligned = self.whisperx.align(
                        result["segments"], self.align_model,
                        self.align_metadata, audio, self.device,
                    )
                    for seg in result_aligned.get("segments", []):
                        for w in seg.get("words", []):
                            words.append({
                                "word": w.get("word", ""),
                                "start": round(w.get("start", 0.0), 3),
                                "end": round(w.get("end", 0.0), 3),
                                "score": round(w.get("score", 0.0), 4),
                            })
                except Exception as e:
                    logger.warning("Alignment failed for %s: %s", audio_path.name, e)

            return {"text": text, "words": words}
        except Exception as e:
            logger.exception("WhisperX failed on %s: %s", audio_path.name, e)
            return None


def validate_dialogue_turns(
    dialogue_id: str,
    dialogue_dir: Path,
    manifest: dict,
    asr_config: dict,
    session: WhisperXSession | None = None,
) -> dict:
    """Per-turn WhisperX validation with word-level alignment. Returns validation result dict."""
    manifest_turns = manifest["turns"]

    turn_results = {}
    turn_words = {}
    failed_turns = []
    total_wer = 0.0
    total_cer = 0.0
    n_turns = 0

    for mt in manifest_turns:
        turn_key = f"turn_{mt['turn']:02d}_{mt['speaker']}"
        written_ref = mt["text_written"]
        spoken = mt.get("text_spoken", mt["text_written"])

        wav_name = mt.get("wav_file", f"turn_{mt['turn']:02d}_{mt['speaker']}.wav")
        wav_path = dialogue_dir / wav_name

        if not wav_path.exists():
            turn_results[turn_key] = {
                "written": written_ref,
                "spoken_tts": spoken,
                "status": "missing_audio",
            }
            failed_turns.append(turn_key)
            continue

        asr_result = session.transcribe(wav_path) if session else None

        if asr_result is None:
            turn_results[turn_key] = {
                "written": written_ref,
                "spoken_tts": spoken,
                "status": "asr_error",
            }
            failed_turns.append(turn_key)
            continue

        asr_text = asr_result["text"]
        words = asr_result.get("words", [])
        turn_words[turn_key] = words

        wer = compute_wer(spoken, asr_text)
        cer = compute_cer(spoken, asr_text)

        wer_threshold = asr_config.get("wer_threshold", 0.15)
        cer_threshold = asr_config.get("cer_threshold", 0.10)
        turn_pass = wer <= wer_threshold and cer <= cer_threshold and len(words) > 0

        turn_results[turn_key] = {
            "written": written_ref,
            "spoken_tts": spoken,
            "asr_result": asr_text,
            "words": words,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "status": "pass" if turn_pass else "fail",
        }

        if not turn_pass:
            failed_turns.append(turn_key)

        total_wer += wer
        total_cer += cer
        n_turns += 1

    overall_wer = round(total_wer / n_turns, 4) if n_turns else 1.0
    overall_cer = round(total_cer / n_turns, 4) if n_turns else 1.0
    wer_threshold = asr_config.get("wer_threshold", 0.50)
    cer_threshold = asr_config.get("cer_threshold", 0.40)
    status = "pass" if (overall_wer <= wer_threshold and overall_cer <= cer_threshold) else "fail"

    return {
        "dialogue_id": dialogue_id,
        "turns": turn_results,
        "turn_words": turn_words,
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "failed_turns": failed_turns,
        "status": status,
    }


def scrub_failed(dialogue_dir: Path, dialogue_id: str) -> None:
    """Delete manifest + audio for a failed dialogue so Stage 4 re-generates."""
    manifest = dialogue_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()
        logger.info("Scrubbed manifest for %s", dialogue_id)

    # Delete turn WAVs and combined audio
    for f in dialogue_dir.glob("*.wav"):
        f.unlink()
    for f in dialogue_dir.glob("*.mp3"):
        f.unlink()


def build_final_dataset(config: dict, failed_ids: set[str], validation_dir: Path) -> dict:
    """Build Moshi-compatible final dataset from stage4 output with word-level timestamps."""
    output_dir = Path(config["output_dir"])
    audio_dir = output_dir / "stage4_audio"
    final_dir = output_dir / "final"
    data_stereo_dir = final_dir / "data_stereo"
    voices_ref_dir = final_dir / "voices"
    final_dir.mkdir(parents=True, exist_ok=True)
    data_stereo_dir.mkdir(parents=True, exist_ok=True)
    voices_ref_dir.mkdir(parents=True, exist_ok=True)
    # Clean old outputs before rebuild
    for old_file in data_stereo_dir.glob("*"):
        old_file.unlink()
    (final_dir / "dataset.jsonl").unlink(missing_ok=True)

    # Download voices from HuggingFace
    from huggingface_hub import snapshot_download
    hf_repo = config.get("voices_hf_repo", "Revolab/voices")
    logger.info("Downloading voices from %s...", hf_repo)
    hf_cache = Path(snapshot_download(hf_repo, repo_type="dataset"))
    voices_src = hf_cache / "data"
    logger.info("Voices at %s", voices_src)

    copied_voices: set[str] = set()

    dataset_entries = []
    stats = {
        "total_dialogues": 0,
        "total_turns": 0,
        "total_duration_s": 0.0,
        "validated_dialogues": 0,
        "validated_duration_s": 0.0,
        "unvalidated_dialogues": 0,
        "unvalidated_duration_s": 0.0,
        "themes": {},
        "llm_backends": {},
    }

    for dialogue_dir in sorted(audio_dir.iterdir()):
        if not dialogue_dir.is_dir():
            continue

        manifest_path = dialogue_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        dialogue_id = manifest["dialogue_id"]

        if dialogue_id in failed_ids:
            logger.debug("Skipping failed %s in final dataset", dialogue_id)
            continue

        # Load validation data for word-level timestamps
        val_path = validation_dir / f"{dialogue_id}_validation.json"
        word_data = {}
        if val_path.exists():
            with open(val_path) as f:
                val = json.load(f)
                word_data = val.get("turn_words", {})
                # Prefer words_forced (forced alignment) over ASR words
                val_turns = val.get("turns", {})
                for turn_key, td in val_turns.items():
                    wf = td.get("words_forced")
                    if wf:
                        word_data[turn_key] = wf

        # Copy MP3 to final
        src_mp3 = dialogue_dir / "combined.mp3"
        dst_mp3 = data_stereo_dir / f"{dialogue_id}.mp3"
        if src_mp3.exists():
            shutil.copy2(src_mp3, dst_mp3)
        else:
            continue

        # Build companion JSON with transcript + word-level timestamps + situation metadata
        situation_meta = manifest.get("situation", {})
        if not situation_meta.get("theme"):
            situation_meta = {
                "situation_id": manifest.get("situation", {}).get("situation_id", ""),
                "theme": "unknown",
                "domain": "unknown",
            }
        characters = situation_meta.get("characters", {})

        voice_map = {}
        for mt in manifest["turns"]:
            spk = mt.get("speaker", "")
            vid = mt.get("voice_id", "")
            if spk and vid and spk not in voice_map:
                voice_map[spk] = vid

        speakers = {}
        for role in ("agent", "human"):
            char = characters.get(role, {})
            vid = voice_map.get(role, "")
            speakers[role] = {
                "voice_id": vid,
                "name": char.get("name", ""),
                "gender": char.get("gender", ""),
                "role": char.get("role", ""),
                "speaking_style": char.get("speaking_style", ""),
                "voice_ref_path": f"voices/{vid}.wav" if vid else "",
            }
            if vid and vid not in copied_voices:
                src = voices_src / f"{vid}.wav"
                if src.exists():
                    shutil.copy2(src, voices_ref_dir / f"{vid}.wav")
                    copied_voices.add(vid)

        companion = {
            "dialogue_id": dialogue_id,
            "llm_backend": manifest.get("llm_backend", "unknown"),
            "situation": situation_meta,
            "speakers": speakers,
            "turns": [],
        }

        offset = 0.0
        gap_s = config["pipeline"].get("gap_between_turns_s", 0.3)
        for mt in manifest["turns"]:
            turn_key = f"turn_{mt['turn']:02d}_{mt['speaker']}"
            turn_word_list = word_data.get(turn_key, [])

            # Use first word's start time to skip TTS leading silence
            leading_silence = 0.0
            if turn_word_list:
                leading_silence = turn_word_list[0].get("start", 0.0)

            offset_words = [
                {
                    "word": w["word"],
                    "start": round(w["start"] - leading_silence + offset, 3),
                    "end": round(w["end"] - leading_silence + offset, 3),
                    "score": w["score"],
                }
                for w in turn_word_list
            ]

            speech_duration = mt["duration_s"] - leading_silence
            turn_entry = {
                "turn": mt["turn"],
                "speaker": mt["speaker"],
                "channel": "left" if mt["speaker"] == "agent" else "right",
                "text_written": mt["text_written"],
                "text_spoken": mt.get("text_spoken", ""),
                "start_s": round(offset, 3),
                "end_s": round(offset + speech_duration, 3),
                "words": offset_words,
            }
            companion["turns"].append(turn_entry)
            offset += speech_duration + gap_s

        companion_path = data_stereo_dir / f"{dialogue_id}.json"
        with open(companion_path, "w") as f:
            json.dump(companion, f, indent=2, ensure_ascii=False)

        dataset_entries.append({
            "path": f"data_stereo/{dialogue_id}.mp3",
            "duration": round(manifest["total_duration_s"], 2),
            "llm_backend": manifest.get("llm_backend", "unknown"),
            "theme": situation_meta.get("theme", "unknown"),
            "speakers": speakers,
        })

        stats["total_dialogues"] += 1
        stats["total_turns"] += len(manifest["turns"])
        stats["total_duration_s"] += manifest["total_duration_s"]
        if word_data:
            stats["validated_dialogues"] += 1
            stats["validated_duration_s"] += manifest["total_duration_s"]
        else:
            stats["unvalidated_dialogues"] += 1
            stats["unvalidated_duration_s"] += manifest["total_duration_s"]
        theme = situation_meta.get("theme", "unknown")
        stats["themes"][theme] = stats["themes"].get(theme, 0) + 1
        backend = manifest.get("llm_backend", "unknown")
        stats["llm_backends"][backend] = stats["llm_backends"].get(backend, 0) + 1

        # Keep WAV files for potential re-validation

    # Write dataset.jsonl
    jsonl_path = final_dir / "dataset.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")

    stats["total_duration_h"] = round(stats["total_duration_s"] / 3600, 2)
    stats["validated_duration_h"] = round(stats["validated_duration_s"] / 3600, 2)
    stats["unvalidated_duration_h"] = round(stats["unvalidated_duration_s"] / 3600, 2)
    stats_path = final_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: ASR validation + final dataset")
    parser.add_argument(
        "--skip-asr", action="store_true",
        help="Skip WhisperX, just assemble final dataset",
    )
    parser.add_argument(
        "--rebuild-final", action="store_true",
        help="Only rebuild final dataset from existing validations (no ASR, no stubs)",
    )
    parser.add_argument(
        "--scrub-only", action="store_true",
        help="Only scrub previously failed dialogues",
    )
    parser.add_argument(
        "--no-scrub", action="store_true",
        help="Validate but don't delete failed outputs",
    )
    parser.add_argument(
        "--align-model", type=str, default=None,
        help="Alignment model for WhisperX (e.g. mesolitica/wav2vec2-xls-r-300m-mixed)",
    )
    args = parser.parse_args()

    config = load_config()
    asr_config = config.get("asr", {})
    output_dir = Path(config["output_dir"])
    audio_dir = output_dir / "stage4_audio"
    validation_dir = output_dir / "stage5_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        logger.error("Stage 4 output not found. Run 04_generate_audio.py first.")
        sys.exit(1)

    # Fast path: just rebuild final from existing validations
    if args.rebuild_final:
        failed_path = validation_dir / "failed_ids.json"
        failed_ids: set[str] = set()
        if failed_path.exists():
            with open(failed_path) as f:
                failed_ids = set(json.load(f))
        stats = build_final_dataset(config, failed_ids, validation_dir)
        logger.info(
            "Rebuilt final: %d dialogues, %.1fh (validated: %d/%.1fh, unvalidated: %d/%.1fh)",
            stats["total_dialogues"], stats["total_duration_h"],
            stats["validated_dialogues"], stats["validated_duration_h"],
            stats["unvalidated_dialogues"], stats["unvalidated_duration_h"],
        )
        return

    failed_ids: set[str] = set()

    # Purge stale skipped_asr stubs so they get real validation
    if not args.skip_asr:
        purged = 0
        for vf in validation_dir.glob("*_validation.json"):
            with open(vf) as f:
                if json.load(f).get("status") == "skipped_asr":
                    vf.unlink()
                    purged += 1
        if purged:
            logger.info("Purged %d stale skipped_asr validation stubs", purged)

    # Load previously failed IDs if scrub-only mode
    failed_path = validation_dir / "failed_ids.json"
    if args.scrub_only:
        if failed_path.exists():
            with open(failed_path) as f:
                failed_ids = set(json.load(f))
            for did in failed_ids:
                ddir = audio_dir / did
                if ddir.is_dir():
                    scrub_failed(ddir, did)
            logger.info("Scrubbed %d failed dialogues", len(failed_ids))
        else:
            logger.info("No previous failures to scrub")
        return

    # Validate each dialogue
    session = None
    if not args.skip_asr:
        session = WhisperXSession(asr_config, align_model_override=args.align_model)

    validations = []
    for dialogue_dir in sorted(audio_dir.iterdir()):
        if not dialogue_dir.is_dir():
            continue

        manifest_path = dialogue_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        dialogue_id = manifest["dialogue_id"]

        # Skip already-validated dialogues
        val_path = validation_dir / f"{dialogue_id}_validation.json"
        if val_path.exists() and not args.skip_asr:
            logger.debug("Skipping %s (validation exists)", dialogue_id)
            with open(val_path) as f:
                validation = json.load(f)
            validations.append(validation)
            if validation["status"] == "fail":
                failed_ids.add(dialogue_id)
            continue

        if args.skip_asr:
            validation = {
                "dialogue_id": dialogue_id,
                "status": "skipped_asr",
                "failed_turns": [],
            }
        else:
            validation = validate_dialogue_turns(
                dialogue_id, dialogue_dir, manifest, asr_config,
                session=session,
            )

        val_path = validation_dir / f"{dialogue_id}_validation.json"
        with open(val_path, "w") as f:
            json.dump(validation, f, indent=2, ensure_ascii=False)

        validations.append(validation)

        if validation["status"] == "fail":
            failed_ids.add(dialogue_id)
            if not args.no_scrub:
                scrub_failed(dialogue_dir, dialogue_id)
                logger.warning(
                    "FAILED %s: WER=%.3f CER=%.3f — turns: %s (scrubbed)",
                    dialogue_id, validation["overall_wer"],
                    validation["overall_cer"], validation["failed_turns"],
                )

    # Save failed IDs for repair loop
    with open(failed_path, "w") as f:
        json.dump(sorted(failed_ids), f, indent=2)

    # Summary
    passed = sum(1 for v in validations if v["status"] != "fail")
    failed = len(failed_ids)
    skipped = sum(1 for v in validations if v["status"] == "skipped_asr")

    summary = {
        "total_validated": len(validations),
        "passed": passed,
        "failed": failed,
        "skipped_asr": skipped,
    }
    with open(validation_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Validation: %d validated, %d passed, %d failed, %d skipped_asr",
        len(validations), passed, failed, skipped,
    )

    # Build final dataset (excluding failed)
    stats = build_final_dataset(config, failed_ids, validation_dir)
    logger.info(
        "Final dataset: %d dialogues, %d turns, %.1f hours",
        stats["total_dialogues"], stats["total_turns"], stats["total_duration_h"],
    )


if __name__ == "__main__":
    main()
