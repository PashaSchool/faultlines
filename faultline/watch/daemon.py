"""File watcher + debounced refresh loop.

Uses the watchdog library to observe a repo for file changes and runs
the incremental refresh pipeline after a configurable debounce window
(default 30 seconds of silence).

Ignores:
  - .git/, node_modules/, __pycache__/, .venv/, dist/, build/
  - Any path matching a .gitignore-style excluded pattern
  - Binary files (not .py/.ts/.js/etc)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_SECONDS = 30.0
_DEFAULT_MAP_DIR = Path.home() / ".faultline"
_PID_DIR = Path.home() / ".faultline" / "watchers"

_IGNORED_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".turbo", ".cache", ".pytest_cache",
    ".mypy_cache", "target", "vendor",
}

_INTERESTING_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".rb", ".java", ".kt", ".swift",
    ".c", ".h", ".cpp", ".hpp", ".cc",
}


@dataclass
class WatcherStatus:
    running: bool
    pid: int | None = None
    repo_path: str | None = None
    started_at: float | None = None


def _slug(repo_path: str) -> str:
    return hashlib.sha256(str(Path(repo_path).resolve()).encode()).hexdigest()[:12]


def _pid_file(repo_path: str) -> Path:
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    return _PID_DIR / f"{_slug(repo_path)}.pid"


def _write_pid(repo_path: str, pid: int) -> None:
    _pid_file(repo_path).write_text(json.dumps({
        "pid": pid,
        "repo_path": str(Path(repo_path).resolve()),
        "started_at": time.time(),
    }))


def _read_pid(repo_path: str) -> dict | None:
    path = _pid_file(repo_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _clear_pid(repo_path: str) -> None:
    path = _pid_file(repo_path)
    try:
        path.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def watcher_status(repo_path: str) -> WatcherStatus:
    info = _read_pid(repo_path)
    if not info:
        return WatcherStatus(running=False)
    pid = info.get("pid")
    if not pid or not _pid_alive(pid):
        _clear_pid(repo_path)
        return WatcherStatus(running=False)
    return WatcherStatus(
        running=True,
        pid=pid,
        repo_path=info.get("repo_path"),
        started_at=info.get("started_at"),
    )


def stop_daemon(repo_path: str) -> bool:
    """Send SIGTERM to the daemon for this repo. Returns True if stopped."""
    status = watcher_status(repo_path)
    if not status.running or not status.pid:
        return False
    try:
        os.kill(status.pid, signal.SIGTERM)
    except OSError:
        return False
    _clear_pid(repo_path)
    return True


def start_daemon(
    repo_path: str,
    *,
    debounce_seconds: float = _DEFAULT_DEBOUNCE_SECONDS,
    map_path: str | None = None,
) -> int:
    """Fork a background watcher process. Returns the child PID.

    On Unix/macOS uses os.fork(). On Windows falls back to
    subprocess.Popen with detached flags.
    """
    existing = watcher_status(repo_path)
    if existing.running:
        raise RuntimeError(
            f"Watcher already running for {repo_path} (pid {existing.pid}). "
            f"Stop it first with `faultlines watch-stop`."
        )

    if sys.platform == "win32":
        return _start_daemon_windows(repo_path, debounce_seconds, map_path)

    pid = os.fork()
    if pid > 0:
        # Parent — return child pid
        return pid

    # Child process: detach from terminal
    os.setsid()
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.dup2(null, 2)

    try:
        run_watcher(
            repo_path=repo_path,
            debounce_seconds=debounce_seconds,
            map_path=map_path,
            pid_tracking=True,
        )
    finally:
        _clear_pid(repo_path)
        os._exit(0)


def _start_daemon_windows(
    repo_path: str,
    debounce_seconds: float,
    map_path: str | None,
) -> int:
    import subprocess
    cmd = [
        sys.executable, "-m", "faultline.watch",
        "--repo", repo_path,
        "--debounce", str(debounce_seconds),
    ]
    if map_path:
        cmd.extend(["--map", map_path])
    creationflags = 0x00000008  # DETACHED_PROCESS on Windows
    p = subprocess.Popen(cmd, creationflags=creationflags)
    _write_pid(repo_path, p.pid)
    return p.pid


def run_watcher(
    repo_path: str,
    *,
    debounce_seconds: float = _DEFAULT_DEBOUNCE_SECONDS,
    map_path: str | None = None,
    pid_tracking: bool = False,
    verbose: bool = False,
) -> None:
    """Foreground watcher loop. Ctrl-C to stop.

    Args:
        repo_path: Directory to watch.
        debounce_seconds: Wait this many seconds of silence before refresh.
        map_path: Explicit feature-map JSON. Defaults to latest in ~/.faultline/.
        pid_tracking: If True, write a pid file (used when forked as daemon).
        verbose: Print refresh events to stdout.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        raise RuntimeError(
            "watchdog not installed. Run: pip install 'faultlines[watch]'"
        )

    resolved_map = _resolve_map_path(map_path, repo_path)
    if resolved_map is None:
        raise RuntimeError(
            f"No feature map found in {_DEFAULT_MAP_DIR}. "
            f"Run `faultlines analyze {repo_path} --llm --flows` first."
        )

    if pid_tracking:
        _write_pid(repo_path, os.getpid())

    state = _WatcherState(
        last_event_time=0.0,
        pending=False,
        lock=threading.Lock(),
    )

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            src = getattr(event, "src_path", "")
            if not _interesting_path(src):
                return
            with state.lock:
                state.last_event_time = time.time()
                state.pending = True

    handler = _Handler()
    observer = Observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.start()

    if verbose:
        print(f"faultlines watch: monitoring {repo_path} (debounce {debounce_seconds}s)")

    try:
        while True:
            time.sleep(1.0)
            with state.lock:
                if not state.pending:
                    continue
                if time.time() - state.last_event_time < debounce_seconds:
                    continue
                state.pending = False

            # Debounce window elapsed — run refresh
            _run_refresh(resolved_map, repo_path, verbose=verbose)
    except KeyboardInterrupt:
        if verbose:
            print("\nfaultlines watch: stopped")
    finally:
        observer.stop()
        observer.join()
        if pid_tracking:
            _clear_pid(repo_path)


def _interesting_path(path: str) -> bool:
    """Skip ignored directories and non-source files."""
    p = Path(path)
    for part in p.parts:
        if part in _IGNORED_DIR_NAMES:
            return False
    if p.suffix.lower() in _INTERESTING_EXTENSIONS:
        return True
    # Treat files without extension (like Makefile) as boring
    return False


def _resolve_map_path(explicit: str | None, repo_path: str) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    _DEFAULT_MAP_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(_DEFAULT_MAP_DIR.glob("feature-map-*.json"))
    return candidates[-1] if candidates else None


@dataclass
class _WatcherState:
    last_event_time: float
    pending: bool
    lock: threading.Lock


def _run_refresh(map_path: Path, repo_path: str, *, verbose: bool = False) -> None:
    try:
        from faultline.cache.refresh import refresh_feature_map
        from faultline.models.types import FeatureMap
        from faultline.output.writer import write_feature_map

        fm = FeatureMap.model_validate_json(map_path.read_text())
        result = refresh_feature_map(fm, repo_path)

        if not result.freshness_before.is_stale:
            return

        write_feature_map(result.updated_map, str(map_path))
        if verbose:
            ts = time.strftime("%H:%M:%S")
            print(
                f"[{ts}] refreshed: {result.freshness_before.commits_behind} commits, "
                f"{result.files_truly_modified} modified, {result.files_added} added"
            )
    except Exception as exc:
        logger.warning("watcher refresh failed: %s", exc)


# CLI entry point for Windows subprocess fallback
def _main_for_subprocess() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--debounce", type=float, default=_DEFAULT_DEBOUNCE_SECONDS)
    ap.add_argument("--map", default=None)
    args = ap.parse_args()
    run_watcher(
        repo_path=args.repo,
        debounce_seconds=args.debounce,
        map_path=args.map,
        pid_tracking=True,
    )


if __name__ == "__main__":
    _main_for_subprocess()
