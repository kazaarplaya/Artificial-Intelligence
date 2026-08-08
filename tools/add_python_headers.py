#!/usr/bin/env python3

import argparse
import datetime
import tokenize
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
    "node_modules",
}


def create_header(
    file_path: Path,
    repository_root: Path,
    student_name: str,
    student_fan: str,
    licence: str | None,
) -> str:
    relative_path = file_path.relative_to(repository_root)

    lines = [
        f"# Student Name: {student_name}",
        f"# Student FAN: {student_fan}",
        f"# File: {relative_path}",
        f"# Date: {datetime.date.today().strftime('%d-%m-%Y')}",
        "# Description: TODO: Add a brief one-line description.",
        f"# Usage: python {relative_path}",
    ]

    if licence:
        lines.append(f"# Licence: {licence}")

    return "\n".join(lines) + "\n\n"


def has_existing_header(contents: str) -> bool:
    first_lines = contents.splitlines()[:30]

    return any(
        line.strip().lower().startswith("# student name:")
        for line in first_lines
    )


def find_header_insertion_index(lines: list[str]) -> int:
    """
    Insert the header after a shebang or Python encoding declaration.
    """

    index = 0

    if lines and lines[0].startswith("#!"):
        index = 1

    # Python encoding declarations are normally on line 1 or line 2.
    if index < len(lines):
        line = lines[index]
        if "coding:" in line or "coding=" in line:
            index += 1

    return index


def add_header(
    file_path: Path,
    repository_root: Path,
    student_name: str,
    student_fan: str,
    licence: str | None,
    dry_run: bool,
) -> bool:
    try:
        # tokenize.open() respects Python encoding declarations.
        with tokenize.open(file_path) as file:
            contents = file.read()
            encoding = file.encoding
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        print(f"Skipped {file_path}: {error}")
        return False

    if has_existing_header(contents):
        print(f"Already has header: {file_path}")
        return False

    header = create_header(
        file_path=file_path,
        repository_root=repository_root,
        student_name=student_name,
        student_fan=student_fan,
        licence=licence,
    )

    lines = contents.splitlines(keepends=True)
    insertion_index = find_header_insertion_index(lines)

    # Ensure the shebang/encoding line is separated from the new header.
    if insertion_index > 0:
        header = "\n" + header

    updated_contents = (
        "".join(lines[:insertion_index])
        + header
        + "".join(lines[insertion_index:])
    )

    if dry_run:
        print(f"Would update: {file_path}")
        return True

    try:
        file_path.write_text(updated_contents, encoding=encoding)
        print(f"Updated: {file_path}")
        return True
    except OSError as error:
        print(f"Failed to update {file_path}: {error}")
        return False


def should_skip(file_path: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORIES for part in file_path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add student information headers to Python files."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Student name",
    )

    parser.add_argument(
        "--fan",
        required=True,
        help="Student FAN",
    )

    parser.add_argument(
        "--licence",
        default=None,
        help="Optional licence, such as MIT Licence",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be changed without modifying them",
    )

    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent.parent

    lab_directories = sorted(repository_root.glob("Lab*_puja0009"))

    if not lab_directories:
        print(f"No matching lab directories found in {repository_root}")
        return

    updated_count = 0

    for lab_directory in lab_directories:
        source_directory = lab_directory / "Source"

        if not source_directory.is_dir():
            print(
                f"Skipped {lab_directory.name}: "
                f"Source directory does not exist"
            )
            continue

        for file_path in sorted(source_directory.rglob("*.py")):
            if add_header(
                file_path=file_path,
                repository_root=repository_root,
                student_name=args.name,
                student_fan=args.fan,
                licence=args.licence,
                dry_run=args.dry_run,
            ):
                updated_count += 1

    result = "would be updated" if args.dry_run else "updated"
    print(f"\n{updated_count} Python file(s) {result}.")

if __name__ == "__main__":
    main()