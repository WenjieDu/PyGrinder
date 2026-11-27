"""Validate PyGrinder release identity and distribution filenames."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "pygrinder" / "version.py"


def read_version() -> str:
    tree = ast.parse(VERSION_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    raise RuntimeError(f"Could not find __version__ in {VERSION_FILE}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def validate_tag(tag: str, version: str) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise SystemExit(f"Release tag/version mismatch: tag={tag!r}, expected={expected!r}")


def validate_dist(dist: Path, version: str) -> None:
    files = sorted(path.name for path in dist.iterdir() if path.is_file())
    expected = {
        f"pygrinder-{version}-py3-none-any.whl",
        f"pygrinder-{version}.tar.gz",
    }
    if set(files) != expected:
        raise SystemExit(f"Unexpected distribution set: {files!r}; expected {sorted(expected)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--event-sha", default="")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()

    version = read_version()
    validate_tag(args.tag, version)

    if args.event_sha:
        head = git_head()
        if head != args.event_sha:
            raise SystemExit(f"Release event SHA mismatch: HEAD={head}, event={args.event_sha}")

    if args.dist is not None:
        validate_dist(args.dist, version)


if __name__ == "__main__":
    main()
