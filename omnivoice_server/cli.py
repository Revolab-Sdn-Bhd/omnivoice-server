"""CLI entrypoint for omnivoice-server."""

from __future__ import annotations

import argparse
import logging
import socket


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omnivoice-server",
        description="OpenAI-compatible HTTP server for OmniVoice TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--host", default=None, help="Bind host (env: OMNIVOICE_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port (env: OMNIVOICE_PORT)")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
        help="Log level (env: OMNIVOICE_LOG_LEVEL)",
    )

    # Model
    parser.add_argument(
        "--model",
        default=None,
        dest="model_id",
        help="HuggingFace model ID or local path (env: OMNIVOICE_MODEL_ID)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device (env: OMNIVOICE_DEVICE)",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        dest="num_step",
        help="Diffusion steps, 1-64 (env: OMNIVOICE_NUM_STEP)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        dest="guidance_scale",
        help="CFG scale, 0-10 (env: OMNIVOICE_GUIDANCE_SCALE)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        default=None,
        dest="denoise",
        help="Enable denoising (env: OMNIVOICE_DENOISE)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_false",
        dest="denoise",
        help="Disable denoising",
    )
    parser.add_argument(
        "--t-shift",
        type=float,
        default=None,
        dest="t_shift",
        help="Noise schedule shift, 0-2 (env: OMNIVOICE_T_SHIFT)",
    )
    parser.add_argument(
        "--position-temperature",
        type=float,
        default=None,
        dest="position_temperature",
        help="Voice diversity temperature, 0-10 (env: OMNIVOICE_POSITION_TEMPERATURE)",
    )
    parser.add_argument(
        "--class-temperature",
        type=float,
        default=None,
        dest="class_temperature",
        help="Token sampling temperature, 0-2 (env: OMNIVOICE_CLASS_TEMPERATURE)",
    )

    # Inference
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        dest="max_concurrent",
        help="Max simultaneous inferences (env: OMNIVOICE_MAX_CONCURRENT)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        dest="request_timeout_s",
        help="Request timeout in seconds (env: OMNIVOICE_REQUEST_TIMEOUT_S)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=int,
        default=None,
        dest="shutdown_timeout",
        help="Seconds to wait for in-flight requests on shutdown (env: OMNIVOICE_SHUTDOWN_TIMEOUT)",
    )

    # Storage
    parser.add_argument(
        "--profile-dir",
        default=None,
        dest="profile_dir",
        help="Voice profile directory (env: OMNIVOICE_PROFILE_DIR)",
    )

    # Auth
    parser.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help="Bearer token for auth. Empty = no auth (env: OMNIVOICE_API_KEY)",
    )

    # Multi-worker / MPS
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        dest="workers",
        help="Number of worker processes (default: 1, set 4+ for MPS parallel)",
    )
    parser.add_argument(
        "--mps-enabled",
        choices=["auto", "true", "false"],
        default=None,
        dest="mps_enabled",
        help="MPS daemon mode (default: auto)",
    )
    parser.add_argument(
        "--mps-active-thread-pct",
        type=int,
        default=None,
        dest="mps_active_thread_percentage",
        help="GPU compute percentage for MPS (1-100, default: 100)",
    )

    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}

    from .config import Settings

    cfg = Settings(**overrides)

    import sys
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stderr,
    )

    logger = logging.getLogger(__name__)

    if cfg.workers > 1:
        # Multi-worker mode with MPS
        from omnivoice_server.mps import MPSManager
        from omnivoice_server.worker_manager import WorkerManager

        mps_mgr = None
        if cfg.mps_should_enable:
            mps_mgr = MPSManager(
                active_thread_percentage=cfg.mps_active_thread_percentage
            )
            if not mps_mgr.start():
                logger.warning(
                    "MPS failed to start, falling back to single-worker mode"
                )
                cfg = cfg.model_copy(update={"workers": 1})

        if cfg.workers > 1:
            worker_mgr = WorkerManager(
                num_workers=cfg.workers,
                host=cfg.host,
                port=cfg.port,
            )
            fd = worker_mgr.create_shared_socket()

            def worker_main():
                """Each worker: create app, load model via lifespan, serve on inherited socket."""
                import asyncio

                import uvicorn

                from omnivoice_server.app import create_app

                app = create_app(cfg)
                # Slot 0 worker writes VRAM measurement during lifespan (see app.py).
                # Use socket from inherited fd
                sock = socket.socket(
                    fileno=fd, family=socket.AF_INET, type=socket.SOCK_STREAM
                )
                config = uvicorn.Config(app, workers=1, log_level=cfg.log_level)
                server = uvicorn.Server(config=config)
                asyncio.run(server.serve(sockets=[sock]))

            # Fork workers with VRAM guard
            worker_mgr.spawn_with_vram_guard(worker_main)

            mps_active = mps_mgr is not None and mps_mgr.status.value == "running"
            logger.info("OMNIVOICE_READY workers=%d mps=%s", cfg.workers, mps_active)
            print(f"OMNIVOICE_READY workers={cfg.workers} mps={mps_active}")

            try:
                worker_mgr.monitor(mps_manager=mps_mgr)
            except KeyboardInterrupt:
                pass
            finally:
                worker_mgr.shutdown(timeout=cfg.shutdown_timeout)
                if mps_mgr:
                    mps_mgr.stop()
            return

    # Single-worker mode: existing behavior (uvicorn.run)
    import uvicorn

    from .app import create_app

    app = create_app(cfg)

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        workers=1,
        loop="asyncio",
        timeout_graceful_shutdown=cfg.shutdown_timeout,
    )


if __name__ == "__main__":
    main()
