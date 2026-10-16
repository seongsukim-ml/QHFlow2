# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Code release identity, independent of the model selected by model.version.

Run ``python -m qhflow2_version [--json]`` from src/; no ML dependencies needed.
"""
import argparse
import json
from pathlib import Path
import subprocess

__version__ = "0.2.0-rc.2"
RELEASE_TAG = f"v{__version__}"


def get_code_metadata():
    """Identify the running source, including local modifications when known.

    Source archives retain the code version but have no Git provenance. Do not
    accidentally attribute an unpacked archive to an enclosing repository.
    """
    metadata = {
        "code_version": __version__,
        "release_tag": RELEASE_TAG,
        "git_commit": None,
        "git_dirty": None,
    }
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return metadata
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True,
                stderr=subprocess.DEVNULL, timeout=5,
            ).strip()

        metadata["git_commit"] = git("rev-parse", "HEAD")
        metadata["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        # Git is optional at runtime; never block training in a source archive
        # or a container without a usable repository.
        pass
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="include Git provenance")
    args = parser.parse_args()
    print(json.dumps(get_code_metadata(), indent=2) if args.json else __version__)


if __name__ == "__main__":
    main()
