"""
Disk-based response cache for repeated synthesis requests.

Stores encoded audio bytes keyed by a hash of all synthesis parameters.
LRU eviction when total size exceeds configured limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU disk cache for synthesis responses."""

    def __init__(self, cache_dir: Path, max_size_gb: float = 5.0) -> None:
        self._dir = cache_dir
        self._max_size_bytes = int(max_size_gb * 1024 ** 3)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_key(**params) -> str:
        """Deterministic cache key from synthesis parameters."""
        payload = json.dumps(params, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def get(self, key: str) -> tuple[bytes, dict] | None:
        """Return (audio_bytes, metadata) on hit, None on miss."""
        bin_path = self._dir / f"{key}.bin"
        meta_path = self._dir / f"{key}.meta"

        if not bin_path.exists():
            with self._lock:
                self._misses += 1
            return None

        data = bin_path.read_bytes()
        os.utime(bin_path, None)  # touch for LRU

        metadata = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        with self._lock:
            self._hits += 1
        logger.debug("Cache hit: %s (%d bytes)", key[:12], len(data))
        return data, metadata

    def put(self, key: str, data: bytes, metadata: dict | None = None) -> None:
        """Store audio bytes + metadata in cache."""
        bin_path = self._dir / f"{key}.bin"
        meta_path = self._dir / f"{key}.meta"

        tmp = bin_path.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(bin_path)
            if metadata:
                meta_tmp = meta_path.with_suffix(".mtmp")
                meta_tmp.write_text(json.dumps(metadata))
                meta_tmp.replace(meta_path)
            if metadata and "text" in metadata:
                txt_path = self._dir / f"{key}.txt"
                txt_path.write_text(metadata["text"], encoding="utf-8")
            logger.debug("Cache put: %s (%d bytes)", key[:12], len(data))
        except OSError as e:
            logger.warning("Cache write failed: %s", e)

        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        try:
            bins = list(self._dir.glob("*.bin"))
            total = sum(f.stat().st_size for f in bins)
        except OSError:
            return

        if total <= self._max_size_bytes:
            return

        files = sorted(bins, key=lambda f: f.stat().st_atime)
        for f in files:
            if total <= self._max_size_bytes * 0.8:
                break
            try:
                size = f.stat().st_size
                f.unlink()
                for ext in (".meta", ".txt"):
                    companion = f.with_suffix(ext)
                    if companion.exists():
                        companion.unlink()
                total -= size
            except OSError:
                pass

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            hits, misses = self._hits, self._misses

        try:
            bins = list(self._dir.glob("*.bin"))
            size_bytes = sum(f.stat().st_size for f in bins)
            file_count = len(bins)
        except OSError:
            size_bytes = 0
            file_count = 0

        return {
            "response_cache_hits": hits,
            "response_cache_misses": misses,
            "response_cache_hit_rate": round(hit_rate, 3),
            "response_cache_size_mb": round(size_bytes / 1024 / 1024, 1),
            "response_cache_entries": file_count,
        }
