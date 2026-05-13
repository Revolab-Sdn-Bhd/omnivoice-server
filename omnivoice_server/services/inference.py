"""
Runs model.generate() in a thread pool with concurrency limiting and
post-request memory cleanup.

DESIGN NOTE — upstream isolation:
  All kwargs construction for model.generate() is centralised in
  OmniVoiceAdapter._build_kwargs(). When OmniVoice adds / renames params,
  only that one method changes — not SynthesisRequest, not the router.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from ..config import Settings
from .model import ModelService

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised when the inference queue is at capacity."""


@dataclass
class SynthesisRequest:
    text: str
    mode: str  # "auto" | "design" | "clone"
    instruct: str | None = None  # for mode="design"
    ref_audio_path: str | None = None  # tmp path, for mode="clone"
    ref_text: str | None = None  # for mode="clone"
    embedding_cache_path: str | None = None  # persistent .pt path for profile embeddings
    speed: float = 1.0
    num_step: int | None = None  # None → use server default
    # Advanced passthrough — None means "use upstream default"
    guidance_scale: float | None = None
    denoise: bool | None = None
    t_shift: float | None = None
    position_temperature: float | None = None
    class_temperature: float | None = None
    duration: float | None = None  # Fixed output duration in seconds
    language: str | None = None  # Optional language code for multilingual pronunciation
    layer_penalty_factor: float | None = None
    preprocess_prompt: bool | None = None
    postprocess_output: bool | None = None
    audio_chunk_duration: float | None = None
    audio_chunk_threshold: float | None = None


@dataclass
class SynthesisResult:
    tensors: list  # list[torch.Tensor], each (1, T)
    duration_s: float
    latency_s: float


class OmniVoiceAdapter:
    """
    Thin adapter that translates SynthesisRequest → model.generate() kwargs.

    WHY THIS EXISTS:
    OmniVoice.generate() accepts ~10 parameters (num_step, speed, instruct,
    ref_audio, ref_text, guidance_scale, denoise, duration, …). As upstream
    adds / renames parameters, only this class needs to change — not the
    request schema, not the router, not the tests.

    This is the single seam between omnivoice-server and the upstream library.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._clone_prompt_cache: dict[str, object] = {}
        self._cache_lock = threading.Lock()

    def build_kwargs(self, req: SynthesisRequest, model) -> dict:
        """Return kwargs dict ready to pass to model.generate()."""
        num_step = req.num_step or self._cfg.num_step
        guidance_scale = (
            req.guidance_scale if req.guidance_scale is not None else self._cfg.guidance_scale
        )
        denoise = req.denoise if req.denoise is not None else self._cfg.denoise
        t_shift = req.t_shift if req.t_shift is not None else self._cfg.t_shift
        position_temperature = (
            req.position_temperature
            if req.position_temperature is not None
            else self._cfg.position_temperature
        )
        class_temperature = (
            req.class_temperature
            if req.class_temperature is not None
            else self._cfg.class_temperature
        )

        kwargs: dict = {
            "text": req.text,
            "num_step": num_step,
            "speed": req.speed,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
            "t_shift": t_shift,
            "position_temperature": position_temperature,
            "class_temperature": class_temperature,
        }

        # Add optional duration parameter if provided
        if req.duration is not None:
            kwargs["duration"] = req.duration

        # Add optional language parameter if provided
        if req.language is not None:
            kwargs["language"] = req.language

        if req.layer_penalty_factor is not None:
            kwargs["layer_penalty_factor"] = req.layer_penalty_factor
        if req.preprocess_prompt is not None:
            kwargs["preprocess_prompt"] = req.preprocess_prompt
        if req.postprocess_output is not None:
            kwargs["postprocess_output"] = req.postprocess_output
        if req.audio_chunk_duration is not None:
            kwargs["audio_chunk_duration"] = req.audio_chunk_duration
        if req.audio_chunk_threshold is not None:
            kwargs["audio_chunk_threshold"] = req.audio_chunk_threshold

        if req.mode == "design" and req.instruct:
            kwargs["instruct"] = req.instruct
        elif req.mode == "clone" and req.ref_audio_path:
            prompt = self._get_or_create_clone_prompt(req, model)
            kwargs["voice_clone_prompt"] = prompt

        return kwargs

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest entries. Must be called with self._cache_lock held."""
        while len(self._clone_prompt_cache) > 32:
            oldest_key = next(iter(self._clone_prompt_cache))
            del self._clone_prompt_cache[oldest_key]

    def _get_or_create_clone_prompt(self, req: SynthesisRequest, model):
        """Return voice_clone_prompt, using a three-tier cache:
        1. In-memory by path key (profile requests, survives re-uploads)
        2. Disk (.pt file next to profile audio, survives restarts)
        3. In-memory by audio content hash (one-shot clone uploads)
        """
        import pathlib

        # Tier 1 & 2: profile-based persistent cache
        path_key = f"path:{req.embedding_cache_path}" if req.embedding_cache_path else None
        if path_key:
            with self._cache_lock:
                if path_key in self._clone_prompt_cache:
                    logger.debug("Reusing in-memory speaker embedding for profile")
                    return self._clone_prompt_cache[path_key]

            disk_path = pathlib.Path(req.embedding_cache_path)  # type: ignore[arg-type]
            if disk_path.exists():
                try:
                    prompt = torch.load(disk_path, weights_only=False)
                    with self._cache_lock:
                        self._clone_prompt_cache[path_key] = prompt
                        self._evict_cache_if_needed()
                    logger.info("Loaded speaker embedding from disk: %s", disk_path.name)
                    return prompt
                except Exception as e:
                    logger.warning("Failed to load cached embedding %s: %s", disk_path.name, e)

        # Tier 3: content-hash in-memory cache (one-shot uploads)
        audio_bytes = pathlib.Path(req.ref_audio_path).read_bytes()  # type: ignore[arg-type]
        content_key = hashlib.sha256(audio_bytes).hexdigest()

        with self._cache_lock:
            if content_key in self._clone_prompt_cache:
                logger.debug("Reusing cached voice clone prompt")
                return self._clone_prompt_cache[content_key]

        logger.info("Encoding voice clone prompt (new ref audio, %d bytes)", len(audio_bytes))
        prompt = model.create_voice_clone_prompt(
            req.ref_audio_path,
            ref_text=req.ref_text,
        )

        # Persist to disk so next restart skips re-encoding
        if req.embedding_cache_path:
            try:
                disk_path = pathlib.Path(req.embedding_cache_path)
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prompt, disk_path)
                logger.info("Saved speaker embedding to disk: %s", disk_path.name)
            except Exception as e:
                logger.warning("Failed to persist embedding: %s", e)

        with self._cache_lock:
            self._clone_prompt_cache[content_key] = prompt
            if path_key:
                self._clone_prompt_cache[path_key] = prompt
            self._evict_cache_if_needed()

        return prompt

    def call(self, req: SynthesisRequest, model) -> list[torch.Tensor]:
        """Call model.generate() and return raw tensors."""
        kwargs = self.build_kwargs(req, model)
        try:
            return model.generate(**kwargs)
        except TypeError as exc:
            # Upstream renamed or removed a param — try graceful fallback
            # by stripping unknown kwargs one-by-one.
            logger.warning(
                f"model.generate() raised TypeError: {exc}. "
                "Attempting fallback with minimal kwargs."
            )
            minimal = {
                "text": kwargs["text"],
                "num_step": kwargs.get("num_step", 32),
            }
            if "instruct" in kwargs:
                minimal["instruct"] = kwargs["instruct"]
            if "voice_clone_prompt" in kwargs:
                minimal["voice_clone_prompt"] = kwargs["voice_clone_prompt"]
            return model.generate(**minimal)


class InferenceService:
    def __init__(
        self,
        model_svc: ModelService,
        cfg: Settings,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._model_svc = model_svc
        self._cfg = cfg
        self._executor = executor
        self._semaphore = (
            asyncio.Semaphore(cfg.max_concurrent) if executor else None
        )
        self._adapter = OmniVoiceAdapter(cfg)

        # Request batching state
        self._batch_queue: asyncio.Queue[tuple[SynthesisRequest, asyncio.Future]] | None = None
        self._batch_task: asyncio.Task | None = None
        self._pending: int = 0

        # Request deduplication: hash → future for in-flight requests
        self._inflight: dict[str, asyncio.Future] = {}

    def start_batch_scheduler(self) -> None:
        """Start the batch scheduler loop (call after event loop starts)."""
        if not self._cfg.batch_enabled:
            return
        self._batch_queue = asyncio.Queue()
        self._batch_task = asyncio.ensure_future(self._batch_scheduler_loop())
        logger.info(
            "Batch scheduler started (max_size=%d, timeout=%dms)",
            self._cfg.batch_max_size, self._cfg.batch_timeout_ms,
        )

    def stop_batch_scheduler(self) -> None:
        """Stop the batch scheduler."""
        if self._batch_task is not None:
            self._batch_task.cancel()
            self._batch_task = None

    @property
    def pending_count(self) -> int:
        return self._pending

    @staticmethod
    def _request_hash(req: SynthesisRequest) -> str:
        """Deterministic hash of request parameters for dedup."""
        import hashlib
        import json

        payload = json.dumps({
            "t": req.text,
            "m": req.mode,
            "i": req.instruct,
            "rap": req.ref_audio_path,
            "ecp": req.embedding_cache_path,
            "s": req.speed,
            "ns": req.num_step,
            "gs": req.guidance_scale,
            "d": req.denoise,
            "ts": req.t_shift,
            "pt": req.position_temperature,
            "ct": req.class_temperature,
            "dur": req.duration,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    async def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        if self._pending >= self._cfg.max_queue_depth:
            raise QueueFullError(
                f"{self._pending} pending >= {self._cfg.max_queue_depth}"
            )

        # Deduplicate concurrent identical requests
        req_hash = self._request_hash(req)
        if req_hash in self._inflight:
            logger.debug("Dedup: joining in-flight request %s", req_hash[:12])
            return await self._inflight[req_hash]

        self._pending += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._inflight[req_hash] = future

        try:
            if self._cfg.batch_enabled and self._batch_queue is not None:
                result = await self._synthesize_batched(req)
            elif self._executor is not None:
                result = await self._synthesize_threaded(req)
            else:
                result = await self._synthesize_direct(req)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._inflight.pop(req_hash, None)
            self._pending -= 1

    async def _synthesize_batched(self, req: SynthesisRequest) -> SynthesisResult:
        """Submit request to batch queue and wait for result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._batch_queue.put((req, future))  # type: ignore[union-attr]
        return await future

    async def _batch_scheduler_loop(self) -> None:
        """Background loop that collects and processes batches."""
        while True:
            try:
                batch: list[tuple[SynthesisRequest, asyncio.Future]] = []

                # Wait for first request
                first = await self._batch_queue.get()  # type: ignore[union-attr]
                batch.append(first)

                # Collect more until timeout or max size
                deadline = asyncio.get_event_loop().time() + self._cfg.batch_timeout_ms / 1000
                while len(batch) < self._cfg.batch_max_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._batch_queue.get(), timeout=remaining  # type: ignore[union-attr]
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                if len(batch) == 1:
                    # Single request — process directly
                    req, future = batch[0]
                    try:
                        result = await self._synthesize_direct(req)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                else:
                    # Batch multiple requests
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Batch scheduler error")
                await asyncio.sleep(0.1)

    async def _process_batch(
        self, batch: list[tuple[SynthesisRequest, asyncio.Future]]
    ) -> None:
        """Process a batch of requests, grouping by voice to avoid cross-voice mixing."""
        # Group by ref_audio_path (voice) to preserve per-request voice identity
        voice_groups: dict[str, list[tuple[SynthesisRequest, asyncio.Future]]] = {}
        for item in batch:
            req = item[0]
            key = req.ref_audio_path or ""
            voice_groups.setdefault(key, []).append(item)

        # Process each voice group as its own batch
        for group in voice_groups.values():
            await self._process_voice_group(group)

    async def _process_voice_group(
        self, batch: list[tuple[SynthesisRequest, asyncio.Future]]
    ) -> None:
        """Process a batch of requests that share the same voice."""
        texts = [req.text for req, _ in batch]
        first_req = batch[0][0]

        batch_req = SynthesisRequest(
            text=texts,  # type: ignore  # list[str] is valid for model.generate()
            mode=first_req.mode,
            instruct=first_req.instruct,
            ref_audio_path=first_req.ref_audio_path,
            ref_text=first_req.ref_text,
            speed=first_req.speed,
            num_step=first_req.num_step,
            guidance_scale=first_req.guidance_scale,
            denoise=first_req.denoise,
            t_shift=first_req.t_shift,
            position_temperature=first_req.position_temperature,
            class_temperature=first_req.class_temperature,
            duration=first_req.duration,
        )

        try:
            result = await self._synthesize_direct(batch_req)

            if len(batch) == len(result.tensors):
                for i, (req, future) in enumerate(batch):
                    tensor = result.tensors[i]
                    dur = tensor.shape[-1] / 24_000
                    future.set_result(SynthesisResult(
                        tensors=[tensor],
                        duration_s=dur,
                        latency_s=result.latency_s,
                    ))
            else:
                logger.warning(
                    "Batch result count mismatch: %d requests, %d outputs",
                    len(batch), len(result.tensors),
                )
                for req, future in batch:
                    try:
                        r = await self._synthesize_direct(req)
                        future.set_result(r)
                    except Exception as e:
                        future.set_exception(e)

        except Exception as e:
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)

    async def _synthesize_direct(self, req: SynthesisRequest) -> SynthesisResult:
        """Direct inference. Uses dedicated executor if available for CUDA graph affinity."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_sync, req)

    async def _synthesize_threaded(self, req: SynthesisRequest) -> SynthesisResult:
        """Threaded inference with semaphore (single-worker fallback mode)."""
        async with self._semaphore:  # type: ignore[union-attr]
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        self._executor, self._run_sync, req
                    ),
                    timeout=self._cfg.request_timeout_s,
                )
                return result
            except asyncio.TimeoutError:
                raise

    def _run_sync(self, req: SynthesisRequest) -> SynthesisResult:
        """Blocking inference. Runs in thread pool thread."""
        t0 = time.monotonic()
        model = self._model_svc.model

        try:
            tensors = self._adapter.call(req, model)
        finally:
            if self._executor is not None:
                _cleanup_memory(self._cfg.device)

        duration_s = sum(t.shape[-1] for t in tensors) / 24_000
        latency_s = time.monotonic() - t0

        rtf = latency_s / duration_s if duration_s > 0 else float("inf")
        logger.debug(
            f"Synthesized {duration_s:.2f}s audio in {latency_s:.2f}s "
            f"(RTF={rtf:.3f})"
        )
        return SynthesisResult(
            tensors=tensors,
            duration_s=duration_s,
            latency_s=latency_s,
        )


def _cleanup_memory(device: str) -> None:
    """Post-inference memory cleanup."""
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"CUDA cache cleanup failed (non-fatal): {e}")
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"MPS cache cleanup failed (non-fatal): {e}")
