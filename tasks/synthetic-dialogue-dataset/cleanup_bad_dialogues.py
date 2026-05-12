#!/usr/bin/env python3
"""Remove dialogues with unalignable turns from final dataset."""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = Path("/mnt/data/work/omnivoice-server/output/synthetic-dialogue/final")
DATA_DIR = FINAL_DIR / "data_stereo"


def main():
    # Find bad dialogues
    bad_dlgs = set()
    for jf in DATA_DIR.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        for t in data.get("turns", []):
            if not t.get("words"):
                bad_dlgs.add(jf.stem)
                break

    logger.info("Found %d bad dialogues to remove", len(bad_dlgs))

    # Remove bad files
    removed_json = 0
    removed_mp3 = 0
    for dlg_id in sorted(bad_dlgs):
        json_path = DATA_DIR / f"{dlg_id}.json"
        mp3_path = DATA_DIR / f"{dlg_id}.mp3"
        if json_path.exists():
            json_path.unlink()
            removed_json += 1
        if mp3_path.exists():
            mp3_path.unlink()
            removed_mp3 += 1

    logger.info("Removed %d JSONs, %d MP3s", removed_json, removed_mp3)

    # Rebuild dataset.jsonl
    remaining = sorted(DATA_DIR.glob("*.json"))
    logger.info("Rebuilding dataset.jsonl with %d entries...", len(remaining))

    stats = {"total_dialogues": len(remaining), "total_turns": 0, "total_hours": 0.0}
    with open(FINAL_DIR / "dataset.jsonl", "w") as out:
        for jf in remaining:
            with open(jf) as f:
                data = json.load(f)
            total_dur = sum(t.get("end_s", 0) - t.get("start_s", 0) for t in data.get("turns", []))
            stats["total_turns"] += len(data.get("turns", []))
            stats["total_hours"] += total_dur / 3600.0
            mp3_name = jf.stem + ".mp3"
            out.write(json.dumps({"id": jf.stem, "path": f"data_stereo/{mp3_name}"}, ensure_ascii=False) + "\n")

    stats["total_hours"] = round(stats["total_hours"], 1)
    with open(FINAL_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Done: %d dialogues, %d turns, %.1f hours", stats["total_dialogues"], stats["total_turns"], stats["total_hours"])


if __name__ == "__main__":
    main()
