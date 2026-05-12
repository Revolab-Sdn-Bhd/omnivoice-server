"""
Vast.ai PyWorker for OmniVoice TTS Server.

Starts the OmniVoice FastAPI server as a subprocess and proxies
requests through the Vast.ai serverless routing layer.

Environment variables (set via workergroup template):
  OMNIVOICE_MODEL_ID       - HuggingFace model ID (default: Revolab/omnivoice)
  OMNIVOICE_MAX_CONCURRENT - Max parallel inferences (default: 4)
  OMNIVOICE_API_KEY        - Optional auth key (empty = no auth)
"""

from __future__ import annotations

import os
import subprocess
import sys

MODEL_SERVER_URL = "http://0.0.0.0"
MODEL_SERVER_PORT = int(os.environ.get("OMNIVOICE_PORT", "8880"))
MODEL_LOG_FILE = "/tmp/omnivoice-server.log"

try:
    from vastai import Worker, BenchmarkConfig, HandlerConfig, LogActionConfig, WorkerConfig
except ImportError:
    print("ERROR: vastai package not installed. Run: pip install vastai", file=sys.stderr)
    sys.exit(1)


def start_model_server() -> subprocess.Popen:
    """Launch OmniVoice server as a background subprocess."""
    log_f = open(MODEL_LOG_FILE, "a")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "omnivoice_server.cli",
            "--host", "0.0.0.0",
            "--port", str(MODEL_SERVER_PORT),
            "--device", "cuda",
            "--max-concurrent", os.environ.get("OMNIVOICE_MAX_CONCURRENT", "4"),
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    print(f"OmniVoice server started (PID {proc.pid}, log: {MODEL_LOG_FILE})")
    return proc


def benchmark_generator():
    """Generate sample TTS payloads for benchmark-driven autoscaling."""
    import random
    import string

    texts = [
        "Hello, this is a benchmark test for the TTS server.",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to the voice synthesis service. How can I help you today?",
        "Testing one two three, this is a short benchmark prompt.",
        "Artificial intelligence is transforming how we interact with technology every day.",
    ]
    text = random.choice(texts)
    return {
        "model": "omnivoice",
        "input": text,
        "voice": "anwar",
        "response_format": "mp3",
        "speed": 1.0,
    }


def workload_calculator(request_body: dict) -> float:
    """Estimate workload units from request for autoscaling.

    For TTS, text length correlates with GPU compute time.
    Returns approximate character count as workload units.
    """
    input_text = request_body.get("input", "")
    return max(len(input_text), 1)


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url="/health",
    handlers=[
        HandlerConfig(
            route="/v1/audio/speech",
            allow_parallel_requests=True,
            max_queue_time=300.0,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                concurrency=4,
                runs=3,
            ),
            workload_calculator=workload_calculator,
        ),
        HandlerConfig(
            route="/v1/audio/voices",
            allow_parallel_requests=True,
        ),
        HandlerConfig(
            route="/v1/models",
            allow_parallel_requests=True,
        ),
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=True,
            max_queue_time=300.0,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete", "Uvicorn running"],
        on_error=["Error", "Traceback", "CUDA out of memory"],
        on_info=["Loading model", "Warmup complete", "VRAM measurement"],
    ),
)


def main():
    print(f"Starting OmniVoice PyWorker on port {MODEL_SERVER_PORT}")
    Worker(worker_config).run()


if __name__ == "__main__":
    main()
