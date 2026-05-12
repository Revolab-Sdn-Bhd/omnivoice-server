#!/usr/bin/env python3
"""Re-upload cleaned final dataset to HF and delete removed files."""

import logging
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationDelete

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FINAL_DIR = Path("/mnt/data/work/omnivoice-server/output/synthetic-dialogue/final")
DATA_DIR = FINAL_DIR / "data_stereo"
REPO_ID = "Revolab/omnidialog"


def shard_key(filename: str) -> str:
    return filename[:2]


def main():
    api = HfApi()

    logger.info("Listing HF repo files...")
    hf_files = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset") if f.startswith("data_stereo/")]
    hf_stems = {f.split("/")[-1].rsplit(".", 1)[0] for f in hf_files if f.endswith((".json", ".mp3"))}
    logger.info("HF has %d data_stereo files (%d unique stems)", len(hf_files), len(hf_stems))

    local_jsons = {f.stem for f in DATA_DIR.glob("*.json")}
    local_mp3s = {f.stem for f in DATA_DIR.glob("*.mp3")}
    local_stems = local_jsons | local_mp3s
    logger.info("Local has %d JSONs, %d MP3s (%d unique stems)", len(local_jsons), len(local_mp3s), len(local_stems))

    # Find files to delete
    to_delete_stems = hf_stems - local_stems
    to_delete_paths = [f for f in hf_files if f.split("/")[-1].rsplit(".", 1)[0] in to_delete_stems]
    logger.info("Need to delete %d files from HF", len(to_delete_paths))

    # Delete in batches using create_commit
    batch_size = 500
    for i in range(0, len(to_delete_paths), batch_size):
        batch = to_delete_paths[i:i + batch_size]
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        batch_num = i // batch_size + 1
        total_batches = (len(to_delete_paths) + batch_size - 1) // batch_size
        logger.info("Deleting batch %d/%d (%d files)...", batch_num, total_batches, len(batch))
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Remove {len(batch)} unalignable dialogues ({batch_num}/{total_batches})",
        )
        logger.info("  Deleted batch %d", batch_num)

    # Upload files, sharded by 2-char prefix
    shards = {}
    for f in DATA_DIR.iterdir():
        if f.suffix in (".json", ".mp3"):
            sk = shard_key(f.name)
            shards.setdefault(sk, []).append(f)

    logger.info("Uploading %d shards...", len(shards))
    for i, (shard, files) in enumerate(sorted(shards.items())):
        json_count = sum(1 for f in files if f.suffix == ".json")
        mp3_count = sum(1 for f in files if f.suffix == ".mp3")
        logger.info("Shard %d/%d: %s (%d json, %d mp3)", i + 1, len(shards), shard, json_count, mp3_count)

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_shard = Path(tmp) / shard
            tmp_shard.mkdir()
            for f in files:
                (tmp_shard / f.name).symlink_to(f)
            api.upload_folder(
                folder_path=str(tmp_shard),
                path_in_repo=f"data_stereo/{shard}",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
        elapsed = time.time() - t0
        logger.info("  Shard %s done in %.1fs", shard, elapsed)

    # Upload root files
    for fname in ("dataset.jsonl", "stats.json", "README.md"):
        fpath = FINAL_DIR / fname
        if fpath.exists():
            logger.info("Uploading %s...", fname)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=REPO_ID,
                repo_type="dataset",
            )

    logger.info("Upload complete!")


if __name__ == "__main__":
    main()
