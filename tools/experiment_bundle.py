#!/usr/bin/env python3
"""Export and import portable RL dashboard experiment bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "rl-experiment-bundle-v1"
ALLOWED_ROOT_FILES = {"train.log", "run_manifest.json", "audit.json"}
ALLOWED_DIRS = {"rollouts", "diagnosis", "diagnostics", "report_materials", "preflight"}
ARCHIVE_RUNS_PREFIX = PurePosixPath("rl_experiment_bundle/runs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def discover_run_dirs(runs_root: Path) -> list[Path]:
    return sorted({path.parent for path in runs_root.rglob("train.log")})


def collect_files(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if len(relative.parts) == 1 and relative.name in ALLOWED_ROOT_FILES:
            files.append(path)
        elif relative.parts and relative.parts[0] in ALLOWED_DIRS:
            files.append(path)
    return files


def safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {path}")
    return relative


def export_bundle(runs_root: Path, output: Path, selected: list[str] | None, include_all: bool) -> dict:
    runs_root = runs_root.resolve()
    if not runs_root.is_dir():
        raise ValueError(f"runs root does not exist: {runs_root}")
    available = discover_run_dirs(runs_root)
    if include_all:
        run_dirs = available
    else:
        available_set = set(available)
        run_dirs = []
        for relative in selected or []:
            candidate = (runs_root / safe_relative(relative)).resolve()
            if candidate not in available_set:
                raise ValueError(f"run not found or has no train.log: {relative}")
            run_dirs.append(candidate)
        run_dirs.sort()
    if not run_dirs:
        raise ValueError("no runs selected")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rl-bundle-") as temporary:
        staging = Path(temporary) / "rl_experiment_bundle"
        manifest_runs = []
        for run_dir in run_dirs:
            relative_run = run_dir.relative_to(runs_root).as_posix()
            files = []
            for source in collect_files(run_dir):
                relative_file = source.relative_to(run_dir)
                destination = staging / "runs" / relative_run / relative_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                files.append({
                    "path": (Path(relative_run) / relative_file).as_posix(),
                    "size": source.stat().st_size,
                    "sha256": sha256(source),
                })
            manifest_runs.append({"run": relative_run, "files": files})

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "toolkit_revision": git_revision(),
            "source_runs_root": str(runs_root),
            "runs": manifest_runs,
        }
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging, arcname="rl_experiment_bundle")
    return manifest


def import_bundle(archive_path: Path, output: Path) -> dict:
    with tarfile.open(archive_path, "r:gz") as archive:
        regular = [member for member in archive.getmembers() if member.isfile()]
        manifest_member = next(
            (member for member in regular if member.name == "rl_experiment_bundle/manifest.json"), None
        )
        if manifest_member is None:
            raise ValueError("archive has no manifest.json")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise ValueError("cannot read manifest.json")
        manifest = json.load(manifest_stream)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported bundle schema")
        expected = {
            file_info["path"]: file_info
            for run in manifest.get("runs", [])
            for file_info in run.get("files", [])
        }
        seen = set()
        output.mkdir(parents=True, exist_ok=True)
        for member in regular:
            archive_path_parts = safe_relative(member.name)
            try:
                relative = archive_path_parts.relative_to(ARCHIVE_RUNS_PREFIX)
            except ValueError:
                continue
            relative_text = relative.as_posix()
            info = expected.get(relative_text)
            if info is None:
                raise ValueError(f"unlisted archive member: {member.name}")
            destination = (output / relative_text).resolve()
            destination.relative_to(output.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with destination.open("wb") as handle:
                shutil.copyfileobj(stream, handle)
            if destination.stat().st_size != info.get("size") or sha256(destination) != info.get("sha256"):
                raise ValueError(f"checksum mismatch: {member.name}")
            seen.add(relative_text)
        missing = sorted(set(expected) - seen)
        if missing:
            raise ValueError(f"archive is missing {len(missing)} manifest files")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--runs-root", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--run", action="append", dest="runs")
    export_parser.add_argument("--all", action="store_true", dest="include_all")
    import_parser = commands.add_parser("import")
    import_parser.add_argument("--archive", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if not args.include_all and not args.runs:
                parser.error("export requires --all or at least one --run")
            manifest = export_bundle(args.runs_root, args.output, args.runs, args.include_all)
            file_count = sum(len(run["files"]) for run in manifest["runs"])
            print(f"exported {len(manifest['runs'])} runs and {file_count} files to {args.output}")
        else:
            manifest = import_bundle(args.archive, args.output)
            print(f"imported {len(manifest['runs'])} runs to {args.output}")
    except (OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
