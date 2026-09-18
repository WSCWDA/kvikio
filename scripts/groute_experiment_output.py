#!/usr/bin/env python3
"""Persist G-Route experiment results under /mnt/gds/results."""

import contextlib
import json
import re
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOT = Path("/mnt/gds/results")


def run_experiment(main, name):
    safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    if not safe_name:
        raise ValueError("experiment name must contain letters or digits")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    directory = RESULT_ROOT / (
        f"groute_{safe_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    metadata = {
        "started_utc": started,
        "command": [sys.executable, *sys.argv],
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "status": "running",
    }
    metadata_path = directory / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
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
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {directory}")
