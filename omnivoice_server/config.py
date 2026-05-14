"""
Server configuration.
Priority: CLI flags > env vars > defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import platformdirs
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import torch


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMNIVOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = Field(default="127.0.0.1", description="Bind host")
    port: int = Field(default=8880, ge=0, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # Model
    model_id: str = Field(
        default="Revolab/omnivoice",
        description="HuggingFace repo ID or local path",
    )
    model_revision: str = Field(
        default="",
        description="Git revision (branch, tag, commit) to load from HuggingFace",
    )
    model_cache_dir: Path | None = Field(
        default=None,
        description="Override HuggingFace model cache directory",
    )
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    num_step: int = Field(default=16, ge=1, le=64)

    # Optimization
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = Field(
        default="none",
        description=(
            "torch.compile mode for LLM backbone. "
            "'none'=disabled, 'max-autotune'=best perf but slow first compile."
        ),
    )
    compile_cache_dir: Path | None = Field(
        default=None,
        description=(
            "Persistent directory for torch.compile Inductor cache. "
            "Populate via: python scripts/precompile_model.py --cache-dir DIR. "
            "If set, server loads cached kernels instead of recompiling."
        ),
    )
    quantization: Literal["none", "fp8wo", "fp8dq", "int8wo", "int8dq"] = Field(
        default="none",
        description=(
            "TorchAO quantization for LLM backbone. "
            "fp8wo=FP8 weight-only, int8wo=INT8 weight-only. "
            "Requires torchao package and FP8-capable GPU for fp8 options."
        ),
    )

    # Advanced generation params (passed through to OmniVoice.generate())
    # Expose the ones users are likely to tune; leave the rest at upstream defaults.
    guidance_scale: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="CFG scale. Higher = stronger voice conditioning.",
    )
    denoise: bool = Field(
        default=True,
        description="Enable upstream denoising token. Recommended on.",
    )
    t_shift: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,  # Upstream docs don't specify max; allowing up to 2.0 for flexibility
        description="Noise schedule shift. Affects quality/speed tradeoff.",
    )
    position_temperature: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description=(
            "Temperature for mask-position selection. "
            "0=deterministic/greedy, higher=more diversity."
        ),
    )
    class_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Temperature for token sampling at each step. "
            "0=greedy, higher=more randomness."
        ),
    )

    # Inference
    max_concurrent: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Max simultaneous inference calls",
    )
    loudness_target_lufs: float | None = Field(
        default=-16.0,
        ge=-60.0,
        le=0.0,
        description=(
            "Target loudness in LUFS for output normalization. "
            "Set to null to disable. Default -16 LUFS (speech standard)."
        ),
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "Number of worker processes. "
            "Default=1 (single-worker, opt-in for multi-worker)."
        ),
    )
    mps_enabled: Literal["auto", "true", "false"] = Field(
        default="auto",
        description="MPS mode: auto=enable when cuda+workers>1, true=force, false=disable",
    )
    mps_active_thread_percentage: int = Field(
        default=100,
        ge=1,
        le=100,
        description="GPU compute percentage for MPS (1-100)",
    )
    request_timeout_s: int = Field(
        default=120,
        description="Max seconds per synthesis request before 504",
    )
    shutdown_timeout: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Seconds to wait for in-flight requests on shutdown",
    )

    # Voice profiles
    profile_dir: Path = Field(
        default=Path(platformdirs.user_data_dir("omnivoice")) / "profiles",
        description="Directory for saved voice cloning profiles",
    )

    # Voices directory (WAV files for SepBox-compatible voice management)
    voices_hf_repo: str = Field(
        default="Revolab/voices",
        description="HuggingFace dataset repo for voice reference audio",
    )
    voices_dir: Path = Field(
        default=Path(__file__).parent / "static" / "speakers",
        description="Local directory for voice WAV files (auto-populated from HF if empty)",
    )
    voices_revision_hash: str = Field(
        default="",
        description="Resolved HF commit hash for voices dataset (set at startup)",
    )

    # Audio post-processing
    target_lufs: float = Field(
        default=-23.0,
        description="Target loudness (LUFS) for generated audio normalization",
    )

    # Auth
    api_key: str = Field(
        default="",
        description="Optional Bearer token. Empty = no auth.",
    )

    # Batching
    batch_enabled: bool = Field(
        default=False,
        description=(
            "Enable request batching: collect requests for batch_timeout_ms "
            "and process them together in a single batched model call."
        ),
    )
    batch_max_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of requests to batch together.",
    )
    batch_timeout_ms: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Max milliseconds to wait before processing a partial batch.",
    )

    # Streaming
    stream_chunk_max_chars: int = Field(
        default=400,
        description="Max chars per sentence chunk when streaming",
    )

    max_ref_audio_mb: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Max upload size for ref_audio files in megabytes.",
    )

    # Response cache
    response_cache_enabled: bool = Field(
        default=True,
        description="Cache encoded audio for repeated identical requests.",
    )
    response_cache_max_gb: float = Field(
        default=5.0,
        ge=0.0,
        description="Max disk space for response cache (GB).",
    )

    # Tracing
    trace_dir: Path = Field(
        default=Path("traces"),
        description="Save a copy of every generated audio + JSON metadata here for tracing.",
    )

    # Backpressure
    max_queue_depth: int = Field(
        default=64,
        ge=1,
        description="Max pending requests. Excess returns 503.",
    )

    @property
    def mps_should_enable(self) -> bool:
        """Resolve auto/true/false to boolean."""
        if self.mps_enabled == "true":
            return True
        if self.mps_enabled == "false":
            return False
        # auto: enable only when cuda + multi-worker
        return self.device == "cuda" and self.workers > 1

    @property
    def max_ref_audio_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_ref_audio_mb * 1024 * 1024

    @field_validator("device")
    @classmethod
    def resolve_auto_device(cls, v: str) -> str:
        if v != "auto":
            return v
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return appropriate torch dtype for device."""
        import torch

        if self.device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    @property
    def torch_device_map(self) -> str:
        """Map to device string for OmniVoice.from_pretrained()."""
        if self.device == "cuda":
            return "cuda:0"
        return self.device  # "mps" or "cpu"
