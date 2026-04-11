"""Tests for WorkerManager: fork, crash recovery, MPS health, VRAM guard, shutdown."""

from __future__ import annotations

import json
import signal
import time
from unittest.mock import MagicMock, patch

from omnivoice_server.worker_manager import WorkerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(num_workers=4, host="127.0.0.1", port=8880):
    return WorkerManager(num_workers=num_workers, host=host, port=port)


# ---------------------------------------------------------------------------
# test_worker_manager_creation
# ---------------------------------------------------------------------------


def test_worker_manager_creation():
    mgr = _make_manager(num_workers=4, host="0.0.0.0", port=9999)
    assert mgr.num_workers == 4
    assert mgr.host == "0.0.0.0"
    assert mgr.port == 9999
    assert mgr.worker_pids == {}
    assert mgr._worker_main is None


# ---------------------------------------------------------------------------
# test_create_shared_socket
# ---------------------------------------------------------------------------


def test_create_shared_socket():
    mgr = _make_manager(port=0)  # port 0 lets OS pick a free port

    mock_sock = MagicMock()
    mock_sock.fileno.return_value = 42

    with patch(
        "omnivoice_server.worker_manager.socket.socket",
        return_value=mock_sock,
    ):
        fd = mgr.create_shared_socket()

    assert fd == 42
    setsockopt_calls = mock_sock.setsockopt.call_args_list
    # Verify SO_REUSEPORT and SO_REUSEADDR were set to 1
    assert any(c[0][2] == 1 for c in setsockopt_calls)
    mock_sock.bind.assert_called_once_with(("127.0.0.1", 0))
    mock_sock.listen.assert_called_once_with(mgr.num_workers)
    assert mgr._shared_socket is mock_sock


# ---------------------------------------------------------------------------
# test_spawn_workers_stores_worker_main
# ---------------------------------------------------------------------------


def test_spawn_workers_stores_worker_main():
    mgr = _make_manager(num_workers=2)

    def dummy_main():
        pass

    with patch.object(mgr, "_fork_worker") as mock_fork:
        mgr.spawn_workers(dummy_main)

    assert mgr._worker_main is dummy_main
    assert mock_fork.call_count == 2
    mock_fork.assert_any_call(0)
    mock_fork.assert_any_call(1)


# ---------------------------------------------------------------------------
# test_restart_worker_actually_forks
# ---------------------------------------------------------------------------


def test_restart_worker_actually_forks():
    mgr = _make_manager(num_workers=4)
    mgr._worker_main = lambda: None

    with patch("omnivoice_server.worker_manager.os.fork", return_value=9999):
        mgr._restart_worker(slot=2, dead_pid=5555)

    assert mgr.worker_pids.get(2) == 9999


# ---------------------------------------------------------------------------
# test_crash_loop_stops_restart
# ---------------------------------------------------------------------------


def test_crash_loop_stops_restart():
    mgr = _make_manager(num_workers=4)
    mgr._worker_main = lambda: None

    # _restart_worker internally calls _should_restart, which tracks crashes.
    # First 3 calls: crash count <= CRASH_THRESHOLD (3) -> forks new worker.
    # 4th call: crash count = 4 > threshold -> no fork.
    with patch(
        "omnivoice_server.worker_manager.os.fork",
        return_value=9999,
    ) as mock_fork:
        # Crash 1: count=1 <= 3 -> restart
        mgr._restart_worker(0, dead_pid=100)
        assert 0 in mgr.worker_pids
        mgr.worker_pids.clear()

        # Crash 2: count=2 <= 3 -> restart
        mgr._restart_worker(0, dead_pid=200)
        assert 0 in mgr.worker_pids
        mgr.worker_pids.clear()

        # Crash 3: count=3 <= 3 -> restart
        mgr._restart_worker(0, dead_pid=300)
        assert 0 in mgr.worker_pids
        mgr.worker_pids.clear()

        # Crash 4: count=4 > 3 -> NO restart
        mgr._restart_worker(0, dead_pid=400)
        assert 0 not in mgr.worker_pids

    # Should_restart returns False now that we have 4 crashes
    assert mgr._should_restart(0) is False
    # Fork was called exactly 3 times (for crashes 1-3, not 4)
    assert mock_fork.call_count == 3


# ---------------------------------------------------------------------------
# test_graceful_shutdown_sends_sigterm
# ---------------------------------------------------------------------------


def test_graceful_shutdown_sends_sigterm():
    mgr = _make_manager(num_workers=3)
    # Manually populate worker_pids
    mgr.worker_pids = {0: 100, 1: 200, 2: 300}

    with patch("omnivoice_server.worker_manager.os.kill") as mock_kill, \
         patch("omnivoice_server.worker_manager.os.waitpid") as mock_waitpid:
        # Simulate all children cleaned up immediately
        mock_waitpid.return_value = (0, 0)

        # After the first poll round, they should be gone
        def waitpid_side_effect(pid_arg, flags):
            return (pid_arg, 0)

        mock_waitpid.side_effect = waitpid_side_effect
        mgr.shutdown(timeout=5)

    # Verify SIGTERM sent to all 3 PIDs
    sigterm_calls = [c for c in mock_kill.call_args_list if c[0][1] == signal.SIGTERM]
    assert len(sigterm_calls) == 3
    killed_pids = {c[0][0] for c in sigterm_calls}
    assert killed_pids == {100, 200, 300}
    assert mgr.worker_pids == {}


# ---------------------------------------------------------------------------
# test_mps_health_check_failure_triggers_full_restart
# ---------------------------------------------------------------------------


def test_mps_health_check_failure_triggers_full_restart():
    mgr = _make_manager(num_workers=3)

    # Setup: populate workers as if they were forked
    mgr.worker_pids = {0: 10, 1: 20, 2: 30}

    mock_mps = MagicMock()
    mock_mps.is_healthy.return_value = False

    def fake_fork(slot):
        mgr.worker_pids[slot] = 1000 + slot

    with patch.object(mgr, "_fork_worker", side_effect=fake_fork) as mock_fork, \
         patch.object(mgr, "shutdown") as mock_shutdown:
        mgr._handle_mps_failure(mock_mps)

    # Verify shutdown called first
    mock_shutdown.assert_called_once()

    # Verify MPS stop and start called
    mock_mps.stop.assert_called_once()
    mock_mps.start.assert_called_once()

    # Verify all workers re-forked
    assert mock_fork.call_count == 3
    mock_fork.assert_any_call(0)
    mock_fork.assert_any_call(1)
    mock_fork.assert_any_call(2)
    assert mgr.worker_pids == {0: 1000, 1: 1001, 2: 1002}


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_find_slot_returns_correct_slot():
    mgr = _make_manager()
    mgr.worker_pids = {0: 100, 1: 200, 2: 300}
    assert mgr._find_slot(200) == 1
    assert mgr._find_slot(999) is None


def test_should_restart_resets_after_window():
    """Crashes outside the window should not count toward threshold."""
    mgr = _make_manager()

    # Add old crashes that are outside the window
    old_time = time.monotonic() - 120  # 2 minutes ago
    mgr._crash_log[0] = [old_time, old_time, old_time]

    # A new crash should be allowed since old ones are pruned
    assert mgr._should_restart(0) is True
    assert len(mgr._crash_log[0]) == 1  # Only the new crash remains


def test_shutdown_sends_sigkill_on_timeout():
    """Workers that don't exit within timeout get SIGKILL."""
    mgr = _make_manager(num_workers=1)
    mgr.worker_pids = {0: 500}

    with patch("omnivoice_server.worker_manager.os.kill") as mock_kill, \
         patch("omnivoice_server.worker_manager.os.waitpid") as mock_waitpid:
        # waitpid never reports exit during the poll phase
        mock_waitpid.return_value = (0, 0)

        # But on SIGKILL path, waitpid succeeds
        def waitpid_side_effect(pid, flags):
            if flags == 0:
                return (pid, 0)
            return (0, 0)

        mock_waitpid.side_effect = waitpid_side_effect
        mgr.shutdown(timeout=1)

    # Should have sent SIGTERM then SIGKILL
    kill_calls = mock_kill.call_args_list
    assert any(c[0][1] == signal.SIGTERM for c in kill_calls)
    assert any(c[0][1] == signal.SIGKILL for c in kill_calls)
    assert mgr.worker_pids == {}


def test_spawn_with_vram_guard_reduces_workers(tmp_path):
    """VRAM guard should reduce num_workers if VRAM is insufficient."""
    from omnivoice_server import worker_manager

    measurement_file = str(tmp_path / "vram.json")
    mgr = _make_manager(num_workers=8)

    # Override the measurement file path
    original_path = worker_manager.VRAM_MEASUREMENT_FILE
    worker_manager.VRAM_MEASUREMENT_FILE = measurement_file
    try:
        measurement = {"total_vram_mb": 10000, "peak_vram_mb": 4000}
        # With headroom 2.0: 10000 / (4000 * 2.0) = 1.25 -> int -> 1
        fork_count = 0

        def fake_fork(slot):
            nonlocal fork_count
            fork_count += 1
            mgr.worker_pids[slot] = 1000 + slot
            # Worker 0 writes the measurement file
            if slot == 0:
                with open(measurement_file, "w") as f:
                    json.dump(measurement, f)

        def exists_side_effect(path):
            return path == measurement_file

        # Patch OS calls that interact with real process state
        with (
            patch.object(mgr, "_fork_worker", side_effect=fake_fork),
            patch(
                "omnivoice_server.worker_manager.os.waitpid",
                return_value=(0, 0),
            ),
            patch(
                "omnivoice_server.worker_manager.os.path.exists",
                side_effect=exists_side_effect,
            ),
            patch("omnivoice_server.worker_manager.os.remove"),
        ):
            mgr.spawn_with_vram_guard(lambda: None)

        # Should have reduced to 1 worker, so only slot 0 was forked
        assert mgr.num_workers == 1
        assert fork_count == 1
    finally:
        worker_manager.VRAM_MEASUREMENT_FILE = original_path
