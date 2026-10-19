"""Dependency-free HTTP dashboard for persistent RL experiment objects."""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .experiment_store import ExperimentStore


STATIC_DIR = Path(__file__).with_name("web")


class DashboardState:
    def __init__(self, store: ExperimentStore, refresh_seconds: int) -> None:
        self.store = store
        self.refresh_seconds = refresh_seconds
        self.refresh_lock = threading.Lock()
        self.last_refresh = 0.0
        self.last_error: str | None = None

    def refresh(self, force: bool = False) -> list[dict]:
        with self.refresh_lock:
            if not force and time.time() - self.last_refresh < self.refresh_seconds:
                return self.store.list_experiments()
            try:
                result = self.store.refresh_all()
                self.last_error = None
                self.last_refresh = time.time()
                return result
            except Exception as error:  # keep the dashboard readable during partial writes
                self.last_error = str(error)
                return self.store.list_experiments()


def _summary(item: dict) -> dict:
    latest = item.get("latest_metrics", {})
    return {
        "experiment_id": item.get("experiment_id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "started_at": item.get("started_at"),
        "updated_at": item.get("updated_at"),
        "finished_at": item.get("finished_at"),
        "config": item.get("config", {}),
        "comparison_readiness": item.get("comparison_readiness", {}),
        "run_manifest": item.get("run_manifest", {}),
        "manifest_missing": item.get("manifest_missing", []),
        "progress": item.get("progress", {}),
        "labels": item.get("labels", []),
        "latest_metrics": {
            key: latest.get(key) for key in (
                "reward_mean", "reward_min", "reward_max", "reward_std", "valid_ratio", "positive_ratio", "clip_ratio",
                "entropy", "kl_loss", "grad_norm", "rollouts",
            )
        },
    }


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RLExperimentDashboard/0.1"

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path.startswith("/api/"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                return
            relative = "index.html" if path in ("", "/") else path.lstrip("/")
            candidate = (STATIC_DIR / relative).resolve()
            if not candidate.is_file():
                candidate = STATIC_DIR / "index.html"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(candidate.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/api/health":
                self._json({
                    "ok": state.last_error is None,
                    "last_refresh": state.last_refresh,
                    "error": state.last_error,
                })
                return
            if path == "/api/experiments":
                experiments = state.refresh()
                self._json([_summary(item) for item in experiments])
                return
            if path == "/api/refresh":
                experiments = state.refresh(force=True)
                self._json({"count": len(experiments), "refreshed_at": state.last_refresh})
                return
            if path == "/api/compare":
                experiment_ids = [value for value in parse_qs(parsed.query).get("id", []) if value]
                if len(experiment_ids) != 2:
                    self._json({"error": "exactly two experiment ids are required"}, HTTPStatus.BAD_REQUEST)
                    return
                comparison = state.store.compare_experiments(experiment_ids)
                if comparison is None:
                    self._json({"error": "experiment not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(comparison)
                return
            if path.startswith("/api/experiments/"):
                self._experiment_api(path, parse_qs(parsed.query))
                return
            self._static(path)

        def _experiment_api(self, path: str, query: dict[str, list[str]]) -> None:
            suffix = path.removeprefix("/api/experiments/")
            parts = [part for part in suffix.split("/") if part]
            if not parts:
                self._json({"error": "missing experiment id"}, HTTPStatus.BAD_REQUEST)
                return
            experiment_id = parts[0]
            if len(parts) == 1:
                item = state.store.get_experiment(experiment_id)
                if item is None:
                    state.refresh(force=True)
                    item = state.store.get_experiment(experiment_id)
                if item is None:
                    self._json({"error": "experiment not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(item)
                return
            if len(parts) == 2 and parts[1] == "rollouts":
                step = query.get("step", [None])[0]
                limit = query.get("limit", ["12"])[0]
                try:
                    step_value = int(step) if step is not None else None
                    limit_value = min(max(int(limit), 1), 100)
                except ValueError:
                    self._json({"error": "invalid step or limit"}, HTTPStatus.BAD_REQUEST)
                    return
                records = state.store.get_rollouts(experiment_id, step_value, limit_value)
                self._json(records)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _static(self, path: str) -> None:
            relative = "index.html" if path in ("", "/") else path.lstrip("/")
            candidate = (STATIC_DIR / relative).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                candidate = STATIC_DIR / "index.html"
            content = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            content = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def _refresh_loop(state: DashboardState, stop: threading.Event) -> None:
    while not stop.wait(state.refresh_seconds):
        state.refresh(force=True)


def serve(
    runs_root: str,
    state_dir: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_seconds: int = 10,
    stale_seconds: int = 300,
) -> None:
    store = ExperimentStore(runs_root, state_dir, stale_seconds=stale_seconds)
    state = DashboardState(store, refresh_seconds)
    state.refresh(force=True)
    stop = threading.Event()
    worker = threading.Thread(target=_refresh_loop, args=(state, stop), daemon=True)
    worker.start()
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"RL dashboard: http://{host}:{port}")
    print(f"Runs root: {Path(runs_root).resolve()}")
    print(f"State dir: {Path(state_dir).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the RL experiment dashboard")
    parser.add_argument("--runs-root", required=True, help="Root directory containing training runs")
    parser.add_argument("--state-dir", required=True, help="Persistent state directory outside Git")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument("--stale-seconds", type=int, default=300)
    args = parser.parse_args()
    serve(
        runs_root=args.runs_root,
        state_dir=args.state_dir,
        host=args.host,
        port=args.port,
        refresh_seconds=max(args.refresh_seconds, 2),
        stale_seconds=max(args.stale_seconds, 10),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
