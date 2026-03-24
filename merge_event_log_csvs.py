#!/usr/bin/env python3
"""Merge per-annotation event log CSVs into worker and factory CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = DEFAULT_PROJECT_ROOT / "process_mining_event_logs"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "merged_process_mining_event_logs"
BASE_EVENT_DATETIME = datetime(2000, 1, 1, 0, 0, 0)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-annotation CSV event logs into one sequential CSV per worker. "
            "With factory --all, process all workers in the factory and also create "
            "one factory-level concatenated CSV."
        )
    )
    parser.add_argument(
        "factory",
        help="Factory folder name, for example: factory_001",
    )
    parser.add_argument(
        "worker",
        nargs="?",
        help="Worker folder name, for example: worker_001",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Merge all workers under the given factory and also create a factory CSV.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Per-annotation CSV root directory (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Merged CSV output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged CSV files.",
    )
    args = parser.parse_args()

    if args.all:
        if args.worker:
            parser.error("Do not pass a worker when using --all.")
    elif not args.worker:
        parser.error("Provide a worker, or use --all.")

    return args


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def format_timestamp(value: datetime) -> str:
    return value.strftime(TIMESTAMP_FORMAT)


def iter_worker_dirs(factory_dir: Path) -> list[Path]:
    if not factory_dir.is_dir():
        raise FileNotFoundError(f"Factory CSV directory not found: {factory_dir}")
    return sorted(path for path in factory_dir.iterdir() if path.is_dir())


def iter_csv_paths(worker_dir: Path) -> list[Path]:
    return sorted(path for path in worker_dir.glob("*.csv") if path.is_file())


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"CSV file has no header: {csv_path}")
    if "start_timestamp" not in fieldnames or "end_timestamp" not in fieldnames:
        raise ValueError(f"CSV file missing required timestamp columns: {csv_path}")
    return fieldnames, rows


def validate_fieldnames(expected: list[str], actual: list[str], path: Path) -> None:
    if expected != actual:
        raise ValueError(
            f"CSV schema mismatch for {path}.\nExpected: {expected}\nActual:   {actual}"
        )


def merge_worker_csvs(worker_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_paths = iter_csv_paths(worker_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {worker_dir}")

    merged_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    cumulative_offset = timedelta(0)

    for index, csv_path in enumerate(csv_paths):
        current_fieldnames, rows = read_csv_rows(csv_path)
        if fieldnames is None:
            fieldnames = current_fieldnames
        else:
            validate_fieldnames(fieldnames, current_fieldnames, csv_path)

        shifted_rows: list[dict[str, str]] = []
        file_last_end: datetime | None = None
        for row in rows:
            shifted_row = dict(row)
            shifted_start = parse_timestamp(row["start_timestamp"]) + cumulative_offset
            shifted_end = parse_timestamp(row["end_timestamp"]) + cumulative_offset
            shifted_row["start_timestamp"] = format_timestamp(shifted_start)
            shifted_row["end_timestamp"] = format_timestamp(shifted_end)
            shifted_rows.append(shifted_row)

            if file_last_end is None or shifted_end > file_last_end:
                file_last_end = shifted_end

        merged_rows.extend(shifted_rows)

        if file_last_end is not None:
            cumulative_offset = file_last_end - BASE_EVENT_DATETIME
        elif index == 0:
            cumulative_offset = timedelta(0)

    return fieldnames or [], merged_rows


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def worker_output_path(output_root: Path, factory: str, worker: str) -> Path:
    return output_root / factory / f"{worker}.csv"


def factory_output_path(output_root: Path, factory: str) -> Path:
    return output_root / factory / f"{factory}.csv"


def merge_one_worker(
    input_root: Path,
    output_root: Path,
    factory: str,
    worker: str,
    overwrite: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    worker_dir = input_root / factory / worker
    if not worker_dir.is_dir():
        raise FileNotFoundError(f"Worker CSV directory not found: {worker_dir}")

    output_path = worker_output_path(output_root, factory, worker)
    if output_path.exists() and not overwrite:
        print(f"Keeping existing merged CSV {output_path}", flush=True)
        fieldnames, rows = read_csv_rows(output_path)
        return fieldnames, rows

    fieldnames, rows = merge_worker_csvs(worker_dir)
    print(f"Merging {factory}/{worker} -> {output_path}", flush=True)
    write_csv(output_path, fieldnames, rows)
    return fieldnames, rows


def build_factory_rows(worker_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    combined_rows = [dict(row) for row in worker_rows]
    combined_rows.sort(
        key=lambda row: (
            row.get("worker", ""),
            row.get("start_timestamp", ""),
            row.get("end_timestamp", ""),
            row.get("activity", ""),
            row.get("process", ""),
        )
    )
    return combined_rows


def merge_factory_workers(
    input_root: Path,
    output_root: Path,
    factory: str,
    overwrite: bool,
) -> None:
    factory_dir = input_root / factory
    worker_dirs = iter_worker_dirs(factory_dir)
    if not worker_dirs:
        raise FileNotFoundError(f"No worker CSV directories found under {factory_dir}")

    merged_worker_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for worker_dir in worker_dirs:
        worker_fieldnames, worker_rows = merge_one_worker(
            input_root=input_root,
            output_root=output_root,
            factory=factory,
            worker=worker_dir.name,
            overwrite=overwrite,
        )
        if fieldnames is None:
            fieldnames = worker_fieldnames
        else:
            validate_fieldnames(fieldnames, worker_fieldnames, worker_dir)
        merged_worker_rows.extend(worker_rows)

    output_path = factory_output_path(output_root, factory)
    if output_path.exists() and not overwrite:
        print(f"Keeping existing factory CSV {output_path}", flush=True)
        return

    factory_rows = build_factory_rows(merged_worker_rows)
    print(f"Creating factory CSV {output_path}", flush=True)
    write_csv(output_path, fieldnames or [], factory_rows)


def main() -> int:
    args = parse_args()

    print(f"Input root: {args.input_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)

    if args.all:
        merge_factory_workers(
            input_root=args.input_root,
            output_root=args.output_root,
            factory=args.factory,
            overwrite=args.overwrite,
        )
    else:
        merge_one_worker(
            input_root=args.input_root,
            output_root=args.output_root,
            factory=args.factory,
            worker=args.worker,
            overwrite=args.overwrite,
        )

    print("Finished.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
