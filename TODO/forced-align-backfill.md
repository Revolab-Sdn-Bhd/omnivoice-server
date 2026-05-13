# TODO: Backfill words_forced on all validation files

**Script:** `tasks/synthetic-dialogue-dataset/backfill_forced_align.py`
**Status:** Pending — run after pipeline reaches 1000h target
**Scale:** 17,315 files / 199,329 turns need `words_forced` backfilled

## Why

The forced alignment feature (`words_forced`) was added after most of the dataset was already validated. Only 66 turns have it. All existing validation files need it backfilled so word-level timestamps use the original input text instead of ASR-detected words.

## When

After main pipeline completes (~495h/1000h currently). Backfill is GPU-intensive (WhisperX inference per turn) and would compete with the pipeline.

## Command

```bash
# Dry run first
uv run python tasks/synthetic-dialogue-dataset/backfill_forced_align.py --dry-run

# Full backfill (resumable — skips turns that already have words_forced)
uv run python tasks/synthetic-dialogue-dataset/backfill_forced_align.py

# With custom alignment model
uv run python tasks/synthetic-dialogue-dataset/backfill_forced_align.py \
  --align-model mesolitica/wav2vec2-xls-r-300m-mixed

# After backfill, rebuild final dataset to pick up new words_forced
uv run python tasks/synthetic-dialogue-dataset/05_validate_asr.py --rebuild-final
```

## Notes

- Script is resumable — safe to interrupt and restart
- After backfill, must run `--rebuild-final` to propagate `words_forced` into final dataset JSONs
- Estimated time: many hours for 200K turns on a single GPU
