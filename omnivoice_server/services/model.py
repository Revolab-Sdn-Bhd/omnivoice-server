"""
Loads and holds the OmniVoice model singleton.
Model is loaded once at startup; never reloaded during runtime.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import psutil
import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice

from ..config import Settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None
        self._loaded = False
        self.model_revision_hash: str = ""

    async def load(self) -> None:
        """Load model in a thread (blocking op, must not block event loop)."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            await loop.run_in_executor(ex, self._load_sync)

    async def reload(self) -> None:
        """Unload current model and reload (e.g. after revision change)."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            await loop.run_in_executor(ex, self._unload_sync)
            await loop.run_in_executor(ex, self._load_sync)

    def _unload_sync(self) -> None:
        """Free current model from GPU/CPU memory."""
        if self._model is not None:
            logger.info("Unloading current model...")
            del self._model
            self._model = None
            self._loaded = False
        gc.collect()
        if self.cfg.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_sync(self) -> None:
        from omnivoice import OmniVoice

        ram_before = _get_ram_mb()
        t0 = time.monotonic()

        logger.info(f"Loading model '{self.cfg.model_id}' on {self.cfg.device}...")

        for dtype in self._dtype_candidates():
            try:
                from_pretrained_kwargs = {
                    "device_map": self.cfg.torch_device_map,
                    "dtype": dtype,
                }
                if self.cfg.model_cache_dir is not None:
                    from_pretrained_kwargs["cache_dir"] = str(self.cfg.model_cache_dir)
                if self.cfg.model_revision:
                    from_pretrained_kwargs["revision"] = self.cfg.model_revision
                model = OmniVoice.from_pretrained(
                    self.cfg.model_id,
                    **from_pretrained_kwargs,
                )
                test = model.generate(text="test", num_step=4)
                if self._has_nan(test):
                    logger.warning(f"dtype={dtype} produced NaN, trying next...")
                    del model
                    gc.collect()
                    continue
                self._model = model
                break
            except Exception as e:
                logger.warning(f"Failed to load with dtype={dtype}: {e}")
                continue

        if self._model is None:
            raise RuntimeError(
                f"Failed to load OmniVoice on device={self.cfg.device}. "
                "Try --device cpu or check GPU/MPS availability."
            )

        elapsed = time.monotonic() - t0
        ram_after = _get_ram_mb()
        logger.info(
            f"Model loaded in {elapsed:.1f}s. "
            f"RAM: {ram_before:.0f}MB -> {ram_after:.0f}MB "
            f"(+{ram_after - ram_before:.0f}MB)"
        )

        # Apply post-load optimizations
        self._apply_optimizations()

        self._resolve_revision_hash()

        self._loaded = True

    def _dtype_candidates(self) -> list:
        if self.cfg.device in ("cuda", "mps"):
            return [torch.float32, torch.float16, torch.bfloat16]
        return [torch.float32]

    @staticmethod
    def _has_nan(tensors: list) -> bool:
        import numpy as np

        def _check(v):
            if isinstance(v, torch.Tensor):
                return torch.isnan(v).any().item()
            if isinstance(v, np.ndarray):
                return bool(np.isnan(v).any())
            return False

        return any(_check(t) for t in tensors)

    @property
    def model(self) -> OmniVoice:
        if not self._loaded:
            raise RuntimeError("Model not loaded yet")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _resolve_revision_hash(self) -> None:
        """Resolve the loaded model's commit hash."""
        # If a specific revision was requested, use it directly
        if self.cfg.model_revision:
            self.model_revision_hash = self.cfg.model_revision[:8]
            return

        # Otherwise resolve from HF cache (picks latest cached)
        try:
            from huggingface_hub import scan_cache_dir
            repo_id = self.cfg.model_id
            cache = scan_cache_dir(self.cfg.model_cache_dir) if self.cfg.model_cache_dir else scan_cache_dir()
            for repo in cache.repos:
                if repo.repo_id == repo_id:
                    rev = next(iter(repo.revisions))
                    self.model_revision_hash = rev.commit_hash[:8]
                    return
        except Exception as e:
            logger.debug(f"Could not resolve model revision hash: {e}")

    def _apply_optimizations(self) -> None:
        """Apply torch.compile and/or TorchAO quantization to the LLM backbone."""
        if self._model is None:
            return

        # Quantization first (operates on raw weights)
        if self.cfg.quantization != "none":
            self._apply_quantization()

        # Then torch.compile (works on quantized or normal model)
        if self.cfg.compile_mode != "none":
            self._apply_compile()

    def _apply_quantization(self) -> None:
        """Apply TorchAO quantization to the LLM backbone."""
        from torchao.quantization import quantize_

        configs = {
            "fp8wo": "Float8WeightOnlyConfig",
            "fp8dq": "Float8DynamicActivationFloat8WeightConfig",
            "int8wo": "Int8WeightOnlyConfig",
            "int8dq": "Int8DynamicActivationInt8WeightConfig",
        }
        import torchao.quantization as tq

        cls_name = configs[self.cfg.quantization]
        config_cls = getattr(tq, cls_name)
        config = config_cls()

        logger.info("Applying TorchAO quantization=%s to LLM backbone", self.cfg.quantization)
        assert self._model is not None
        quantize_(self._model.llm, config=config)
        logger.info("Quantization applied")

    def _apply_compile(self) -> None:
        """Apply torch.compile to the LLM backbone."""
        import os

        # Set persistent cache dir so compiled kernels survive restarts
        if self.cfg.compile_cache_dir is not None:
            cache_dir = str(self.cfg.compile_cache_dir)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            logger.info("Inductor cache dir: %s", cache_dir)

        logger.info(
            "Applying torch.compile(mode=%s) to LLM backbone", self.cfg.compile_mode
        )
        assert self._model is not None
        compiled = torch.compile(self._model.llm, mode=self.cfg.compile_mode)
        self._model.llm = compiled

        cache_status = ""
        if self.cfg.compile_cache_dir is not None:
            cache_status = " (cached kernels will be reused if available)"
        logger.info(
            "torch.compile applied%s — first inference triggers compilation",
            cache_status,
        )


def _get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024
