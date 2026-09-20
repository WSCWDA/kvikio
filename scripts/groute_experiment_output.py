#!/usr/bin/env python3
"""Persist G-Route experiment results under /mnt/gds/results."""

import contextlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOT = Path("/mnt/gds/results")


def _consume_flag(name):
    """Remove a wrapper-owned flag from argv and return its value, if present."""
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


def _next_run_directory(experiment_directory):
    experiment_directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        candidate = experiment_directory / f"run_{number:02d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"no available run directory under {experiment_directory}")


def run_experiment(main, name):
    safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    if not safe_name:
        raise ValueError("experiment name must contain letters or digits")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    requested_directory = _consume_flag("--output-dir")
    if requested_directory is None:
        directory = _next_run_directory(RESULT_ROOT / f"groute_{safe_name}")
    else:
        directory = Path(requested_directory).expanduser().resolve()
        try:
            directory.relative_to(RESULT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"result directory must be below {RESULT_ROOT}") from error
        if not any(part.startswith("groute") for part in directory.parts):
            raise ValueError("result directory must be inside a groute* directory")
        directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    metadata = {
        "started_utc": started,
        "command": [sys.executable, *sys.argv],
        "output_directory": str(directory),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status": "running",
    }
    metadata_path = directory / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    previous_result_dir = os.environ.get("GROUTE_RESULT_DIR")
    os.environ["GROUTE_RESULT_DIR"] = str(directory)
    try:
        with (directory / "results.jsonl").open("w", encoding="utf-8") as output:
            with contextlib.redirect_stdout(output):
                main()
    except BaseException:
        metadata["status"] = "failed"
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    else:
        metadata["status"] = "completed"
    finally:
        if previous_result_dir is None:
            os.environ.pop("GROUTE_RESULT_DIR", None)
        else:
            os.environ["GROUTE_RESULT_DIR"] = previous_result_dir
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {directory}")
