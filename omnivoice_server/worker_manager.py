"""Worker process manager with crash recovery, MPS health monitoring, and VRAM guard."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

CRASH_THRESHOLD = 3
CRASH_WINDOW_S = 60
MPS_CHECK_INTERVAL_S = 30
VRAM_HEADROOM_FACTOR = 2.0
VRAM_MEASUREMENT_FILE = "/tmp/omnivoice_vram_measurement.json"


class WorkerManager:
    """Forks and supervises worker processes with crash recovery and MPS awareness."""

    def __init__(self, num_workers: int, host: str, port: int) -> None:
        self.num_workers = num_workers
        self.host = host
        self.port = port
        self.worker_pids: dict[int, int] = {}  # slot -> pid
        self._worker_main: Callable | None = None
        self._shared_socket_fd: int | None = None
        self._shared_socket: socket.socket | None = None  # prevent GC of socket object
        self._crash_log: dict[int, list[float]] = {}  # slot -> list of timestamps
        self._monitoring = False

    def create_shared_socket(self) -> int:
        """Create a TCP socket with SO_REUSEPORT. Returns the file descriptor.

        Keeps a reference to the socket object to prevent garbage collection
        which would close the underlying fd.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(self.num_workers)
        self._shared_socket = sock
        self._shared_socket_fd = sock.fileno()
        logger.info(
            "Shared socket created on %s:%d (fd=%d)",
            self.host,
            self.port,
            self._shared_socket_fd,
        )
        return self._shared_socket_fd

    def spawn_workers(self, worker_main: Callable) -> None:
        """Fork N workers, each calling worker_main() in the child process."""
        self._worker_main = worker_main
        for slot in range(self.num_workers):
            self._fork_worker(slot)

    def _fork_worker(self, slot: int) -> None:
        """Fork a single worker for the given slot.

        In the child: sets OMNIVOICE_WORKER_SLOT env var, calls self._worker_main(),
        then sys.exit(0).
        In the parent: stores the child PID in self.worker_pids[slot].
        """
        pid = os.fork()
        if pid == 0:
            # --- child ---
            os.environ["OMNIVOICE_WORKER_SLOT"] = str(slot)
            try:
                if self._worker_main is not None:
                    self._worker_main()
            except Exception:
                logger.exception("Worker slot %d (pid %d) crashed", slot, os.getpid())
            sys.exit(0)
        else:
            # --- parent ---
            self.worker_pids[slot] = pid
            logger.info("Forked worker slot=%d pid=%d", slot, pid)

    def spawn_with_vram_guard(self, worker_main: Callable) -> None:
        """Spawn workers using VRAM measurement from worker 0 to size the pool.

        Steps:
        1. Delete stale measurement file
        2. Fork worker 0 (which measures VRAM and writes to temp file)
        3. Poll for measurement file (120s timeout, also check worker alive)
        4. Read measurement, calculate safe_workers = total_vram / (peak * VRAM_HEADROOM_FACTOR)
        5. Adjust num_workers if needed
        6. Fork remaining workers
        """
        self._worker_main = worker_main

        # Step 1: Remove stale measurement file
        if os.path.exists(VRAM_MEASUREMENT_FILE):
            os.remove(VRAM_MEASUREMENT_FILE)

        # Step 2: Fork worker 0 (it will write the measurement file)
        self._fork_worker(0)

        # Step 3: Poll for measurement file
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            # Check worker 0 is still alive
            slot0_pid = self.worker_pids.get(0)
            if slot0_pid is not None:
                waited, status = os.waitpid(slot0_pid, os.WNOHANG)
                if waited != 0:
                    logger.error("Worker 0 (pid %d) exited before VRAM measurement", slot0_pid)
                    return

            if os.path.exists(VRAM_MEASUREMENT_FILE):
                break
            time.sleep(0.5)
        else:
            logger.error("Timed out waiting for VRAM measurement from worker 0")
            return

        # Step 4: Read measurement
        with open(VRAM_MEASUREMENT_FILE) as f:
            measurement = json.load(f)

        total_vram_mb = measurement["total_vram_mb"]
        peak_vram_mb = measurement["peak_vram_mb"]
        logger.info(
            "VRAM measurement: total=%.0f MB, peak_per_worker=%.0f MB",
            total_vram_mb,
            peak_vram_mb,
        )

        # Step 5: Calculate safe worker count
        if peak_vram_mb <= 0:
            logger.error("Invalid VRAM measurement: peak_vram_mb=%.0f", peak_vram_mb)
            return

        safe_workers = int(total_vram_mb / (peak_vram_mb * VRAM_HEADROOM_FACTOR))
        safe_workers = max(safe_workers, 1)

        if safe_workers < self.num_workers:
            logger.warning(
                "VRAM guard: reducing workers from %d to %d "
                "(total=%.0f MB, peak=%.0f MB, headroom=%.1fx)",
                self.num_workers,
                safe_workers,
                total_vram_mb,
                peak_vram_mb,
                VRAM_HEADROOM_FACTOR,
            )
            self.num_workers = safe_workers

        # Step 6: Fork remaining workers (slots 1..N-1)
        for slot in range(1, self.num_workers):
            self._fork_worker(slot)

    def monitor(self, mps_manager=None) -> None:
        """Block and monitor worker processes.

        Handles SIGTERM/SIGINT for graceful shutdown.
        Polls os.waitpid(-1, WNOHANG) every 0.5s.
        Checks MPS health every MPS_CHECK_INTERVAL_S.
        Restarts crashed workers within threshold.
        On MPS failure: full restart via _handle_mps_failure().
        """
        self._monitoring = True

        # Install signal handlers
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_signal(signum, frame):
            logger.info("Received signal %d, shutting down workers", signum)
            self._monitoring = False

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        last_mps_check = time.monotonic()

        try:
            while self._monitoring and self.worker_pids:
                # Poll for exited children
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break

                if pid != 0:
                    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                    slot = self._find_slot(pid)
                    if slot is not None:
                        del self.worker_pids[slot]
                        logger.info(
                            "Worker slot=%d pid=%d exited with code %d",
                            slot,
                            pid,
                            exit_code,
                        )
                        self._restart_worker(slot, pid)
                    else:
                        logger.warning("Reaped unknown child pid=%d", pid)

                # Periodic MPS health check
                now = time.monotonic()
                if mps_manager is not None and (now - last_mps_check) >= MPS_CHECK_INTERVAL_S:
                    if not mps_manager.is_healthy():
                        logger.error("MPS health check failed, triggering full restart")
                        self._handle_mps_failure(mps_manager)
                    last_mps_check = now

                time.sleep(0.5)
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)

    def _restart_worker(self, slot: int, dead_pid: int) -> None:
        """Restart a worker in the given slot if within crash threshold."""
        if self._should_restart(slot):
            logger.info("Restarting worker slot=%d", slot)
            self._fork_worker(slot)
        else:
            logger.warning(
                "Worker slot=%d exceeded crash threshold (%d in %ds), not restarting",
                slot,
                CRASH_THRESHOLD,
                CRASH_WINDOW_S,
            )

    def _should_restart(self, slot: int) -> bool:
        """Check if the slot's crash count is within the allowed threshold."""
        now = time.monotonic()
        crashes = self._crash_log.setdefault(slot, [])
        # Prune timestamps outside the window
        crashes[:] = [t for t in crashes if (now - t) < CRASH_WINDOW_S]
        crashes.append(now)
        return len(crashes) <= CRASH_THRESHOLD

    def _handle_mps_failure(self, mps_manager) -> None:
        """Shutdown all workers, restart MPS, re-fork all workers."""
        logger.warning("Handling MPS failure: shutting down all workers")
        self.shutdown()
        logger.info("Restarting MPS daemon")
        mps_manager.stop()
        mps_manager.start()
        logger.info("Re-forking all %d workers", self.num_workers)
        for slot in range(self.num_workers):
            self._fork_worker(slot)

    def shutdown(self, timeout: int = 10) -> None:
        """Graceful shutdown: SIGTERM all workers, wait with deadline, SIGKILL stragglers."""
        if not self.worker_pids:
            return

        pids_to_kill = list(self.worker_pids.values())
        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.worker_pids:
            for slot, pid in list(self.worker_pids.items()):
                try:
                    waited, _status = os.waitpid(pid, os.WNOHANG)
                    if waited != 0:
                        del self.worker_pids[slot]
                        logger.info("Worker slot=%d pid=%d terminated cleanly", slot, pid)
                except ChildProcessError:
                    del self.worker_pids[slot]
            if self.worker_pids:
                time.sleep(0.1)

        # SIGKILL any remaining
        for slot, pid in list(self.worker_pids.items()):
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                logger.warning("Force-killed worker slot=%d pid=%d", slot, pid)
            except (ProcessLookupError, ChildProcessError):
                pass

        self.worker_pids.clear()

    def _find_slot(self, pid: int) -> int | None:
        """Reverse lookup: find the slot number for a given PID."""
        for slot, p in self.worker_pids.items():
            if p == pid:
                return slot
        return None
