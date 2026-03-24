#!/usr/bin/env python3
"""Collate the README and Python source files into a single text file."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Collate README.md and project Python files into one .txt file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "project_collated.txt",
        help="Path to the collated .txt output file.",
    )
    return parser.parse_args()


def iter_python_files(project_root: Path) -> list[Path]:
    python_files = sorted(
        path
        for path in project_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return python_files


def iter_root_txt_files(project_root: Path) -> list[Path]:
    return sorted(
        path
        for path in project_root.glob("*.txt")
        if path.is_file()
    )


def build_collated_text(project_root: Path) -> str:
    paths: list[Path] = []
    readme_path = project_root / "README.md"
    if readme_path.is_file():
        paths.append(readme_path)

    for path in iter_root_txt_files(project_root):
        if path not in paths:
            paths.append(path)

    for path in iter_python_files(project_root):
        if path not in paths:
            paths.append(path)

    sections: list[str] = []
    for path in paths:
        relative_path = path.relative_to(project_root)
        content = path.read_text(encoding="utf-8")
        sections.append(
            "\n".join(
                [
                    f"===== BEGIN FILE: {relative_path} =====",
                    content.rstrip(),
                    f"===== END FILE: {relative_path} =====",
                ]
            )
        )

    return "\n\n".join(sections) + "\n"


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    collated_text = build_collated_text(project_root)
    args.output.write_text(collated_text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
