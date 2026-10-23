"""Experiment provenance (master spec §50).

`git_commit` previously recorded the literal string UNRECORDED so that an
unprovenanced result was visibly unprovenanced. That was the right default but a
poor permanent state; this resolves the real commit when a repository exists and
falls back to an explicit marker when it does not.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path


def git_commit(repo: Path | None = None) -> str:
    root = Path(repo or Path(__file__).resolve().parents[1])
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return "NO_GIT_REPO - results are not tied to a code version"
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # A dirty tree means the commit does NOT describe the code that ran.
        # Flag it rather than silently recording a commit that is a partial lie.
        return f"{commit}{'-DIRTY' if dirty else ''}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"GIT_UNAVAILABLE:{type(exc).__name__}"


def environment_record() -> dict:
    import numpy
    import scipy
    import sklearn
    import tldextract

    return {
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "tldextract": tldextract.__version__,
        "platform": platform.platform(),
    }
