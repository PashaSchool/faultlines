"""File watcher daemon for continuous feature-map refresh.

Background process that listens for file changes in a repo and
triggers incremental refresh after a debounce period. Keeps the
cached feature map always up-to-date so AI agents querying the
MCP server never see stale data.

Public API:
    run_watcher(repo_path, ...) — foreground loop
    start_daemon(repo_path, ...) — background (fork + nohup)
    stop_daemon(repo_path) — kill background watcher

Alternatives:
    cache.auto_refresh — lighter MCP-triggered refresh, no process needed
"""

from faultline.watch.daemon import run_watcher, start_daemon, stop_daemon, watcher_status

__all__ = ["run_watcher", "start_daemon", "stop_daemon", "watcher_status"]
