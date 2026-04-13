"""Pre-compile OmniVoice LLM backbone for production deployment.

Compiles the Qwen3-0.6B LLM backbone with torch.compile and caches
the compiled kernels to a persistent directory. Subsequent server starts
load from cache instead of recompiling.

Usage:
    # Pre-compile with default settings
    CUDA_VISIBLE_DEVICES=3 python scripts/precompile_model.py

    # Pre-compile with specific cache dir
    CUDA_VISIBLE_DEVICES=3 python scripts/precompile_model.py --cache-dir /data/torch_cache

    # Pre-compile and verify
    CUDA_VISIBLE_DEVICES=3 python scripts/precompile_model.py --verify

Output:
    Populates the Inductor cache directory with compiled kernels.
    Server must use the same --compile-mode and --compile-cache-dir
    to pick up the cached compilation.
"""

import argparse
import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sample texts of varying lengths to capture different shape specializations
WARMUP_TEXTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog. This is a test.",
    "This is a longer sentence to ensure the compiler sees different "
    "sequence lengths for proper caching and kernel specialization.",
]


def load_model(device: str, model_id: str):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice

    logger.info("Loading model %s on %s...", model_id, device)
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            model = OmniVoice.from_pretrained(
                model_id,
                device_map=f"{device}:0" if device == "cuda" else device,
                dtype=dtype,
            )
            test = model.generate(text="test", num_step=4)
            if any(torch.isnan(t).any() for t in test):
                del model
                continue
            logger.info("Model loaded with dtype=%s", dtype)
            return model
        except Exception as e:
            logger.warning("Failed with dtype=%s: %s", dtype, e)
    raise RuntimeError("Could not load model")


def precompile(model, compile_mode: str, cache_dir: str):
    """Compile model and populate cache."""
    # Set cache dir before any torch.compile call
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    logger.info("Inductor cache dir: %s", cache_dir)

    # Apply torch.compile to LLM backbone
    logger.info("Applying torch.compile(mode=%s) to LLM backbone...", compile_mode)
    compiled_llm = torch.compile(model.llm, mode=compile_mode)
    model.llm = compiled_llm

    # Warmup with varying text lengths to populate cache for different shapes
    logger.info("Running warmup inferences to populate compile cache...")
    total_start = time.perf_counter()

    for round_num in range(2):
        for i, text in enumerate(WARMUP_TEXTS):
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(text=text, num_step=4)
            elapsed = time.perf_counter() - t0
            dur = output[0].shape[-1] / 24000
            logger.info(
                "  Warmup %d.%d: %.1fs (text=%d chars, audio=%.1fs)",
                round_num + 1, i + 1, elapsed, len(text), dur,
            )

    compile_time = time.perf_counter() - total_start
    logger.info("Warmup + compilation completed in %.1fs", compile_time)

    # Verify cache was populated
    if os.path.exists(cache_dir):
        import subprocess
        result = subprocess.run(["du", "-sh", cache_dir], capture_output=True, text=True)
        logger.info("Cache size: %s", result.stdout.strip())

    return compile_time


def verify(model):
    """Verify compiled model produces correct output."""
    logger.info("Verifying compiled model quality...")
    text = "The quick brown fox jumps over the lazy dog."
    output = model.generate(text=text, num_step=16)
    tensor = output[0]
    dur = tensor.shape[-1] / 24000

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    peak = tensor.abs().max().item()

    logger.info("  Audio: %.2fs, nan=%s, inf=%s, peak=%.4f", dur, has_nan, has_inf, peak)

    if has_nan or has_inf:
        logger.error("FAILED: Output contains NaN or Inf!")
        return False

    logger.info("Verification passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-compile OmniVoice for production")
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compile-mode", default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Inductor cache directory (default: ./torch_compile_cache)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify output after compilation")
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(os.getcwd(), "torch_compile_cache")
    cache_dir = os.path.abspath(cache_dir)

    model = load_model(args.device, args.model_id)
    compile_time = precompile(model, args.compile_mode, cache_dir)

    if args.verify:
        ok = verify(model)
        if not ok:
            logger.error("Verification failed!")
            return 1

    # Benchmark
    logger.info("Running benchmark...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "This is a production test with a longer sentence for TTS benchmarking.",
    ]
    for text in texts:
        t0 = time.perf_counter()
        output = model.generate(text=text, num_step=16)
        elapsed = time.perf_counter() - t0
        dur = output[0].shape[-1] / 24000
        rtf = elapsed / dur if dur > 0 else float("inf")
        logger.info("  '%.40s...' wall=%.3fs dur=%.2fs rtf=%.3f", text, elapsed, dur, rtf)

    print(f"\n{'=' * 60}")
    print("  PRECOMPILATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Compile mode:    {args.compile_mode}")
    print(f"  Compile time:    {compile_time:.1f}s")
    print("")
    print("  To use in production, start server with:")
    print(f"    --compile-mode {args.compile_mode}")
    print(f"    --compile-cache-dir {cache_dir}")
    print("    --num-step 16")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
