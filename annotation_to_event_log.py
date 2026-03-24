#!/usr/bin/env python3
"""Convert raw annotations into process mining CSV event logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "https://api.openai.com/v1"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_ROOT = DEFAULT_PROJECT_ROOT / "raw_transcriptions"
DEFAULT_FACTORY_REPORT_ROOT = DEFAULT_PROJECT_ROOT / "factory_process_mining_reports"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "process_mining_event_logs"
DEFAULT_PROMPT_PATH = DEFAULT_PROJECT_ROOT / "annotation_to_event_log_prompt.txt"
DEFAULT_MODEL = "gpt-5.4"
MAX_CONCURRENT_TASKS = 100

BASE_EVENT_DATETIME = datetime(2000, 1, 1, 0, 0, 0)
CONNECT_TIMEOUT_SECONDS = 30
GENERATE_READ_TIMEOUT_SECONDS = 1800
MAX_REQUEST_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 10.0
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
REPORT_HEADINGS = [
    "Factory Process Overview",
    "Worker Process Summaries",
    "Factory Process Labels",
    "Factory Activity Catalogue",
    "Optimization-Relevant Process Observations",
    "Evidence Limits",
]
CSV_FIELDNAMES = [
    "start_timestamp",
    "end_timestamp",
    "activity",
    "process",
    "factory",
    "worker",
]


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw annotation transcripts into CSV event logs using the "
            "activity/process vocabularies from each factory report. "
            "Reads the OpenAI API key from OPENAI_API_KEY."
        )
    )
    parser.add_argument(
        "factory",
        nargs="?",
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
        help="Process all factories and all workers under the annotation root.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help=f"Raw annotation root directory (default: {DEFAULT_ANNOTATION_ROOT})",
    )
    parser.add_argument(
        "--factory-report-root",
        type=Path,
        default=DEFAULT_FACTORY_REPORT_ROOT,
        help=f"Factory report root directory (default: {DEFAULT_FACTORY_REPORT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"CSV output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"Path to the event-log prompt (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI Responses model to use (default: gpt-5.4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )
    args = parser.parse_args()

    if args.all:
        if args.factory or args.worker:
            parser.error("Do not pass factory/worker when using --all.")
    elif not args.factory or not args.worker:
        parser.error("Provide both factory and worker, or use --all.")

    return args


def read_text_file(path: Path, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"{description} is empty: {path}")
    return content


def raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        message = f"HTTP {response.status_code} for {response.request.method} {response.url}"
        if body:
            message = f"{message}\n{body}"
        raise ApiRequestError(message, status_code=response.status_code) from exc


def should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, ApiRequestError):
        return exc.status_code in TRANSIENT_HTTP_STATUS_CODES
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    api_key: str,
    timeout_seconds: int,
    **kwargs,
) -> dict:
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {api_key}"

    response = session.request(
        method=method,
        url=url,
        headers=headers,
        timeout=(CONNECT_TIMEOUT_SECONDS, timeout_seconds),
        **kwargs,
    )
    raise_for_status(response)
    if not response.content:
        return {}
    return response.json()


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    api_key: str,
    timeout_seconds: int,
    **kwargs,
) -> dict:
    delay_seconds = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return request_json(
                session=session,
                method=method,
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if should_retry_exception(exc) and attempt < MAX_REQUEST_ATTEMPTS:
                print(
                    f"  transient error on {method} {url}: {exc}\n"
                    f"  sleeping {delay_seconds:.0f}s before retry {attempt + 1}/{MAX_REQUEST_ATTEMPTS}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def read_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set or is empty.")
    return api_key


def iter_factory_dirs(annotation_root: Path) -> list[Path]:
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Annotation root directory not found: {annotation_root}")
    return sorted(path for path in annotation_root.iterdir() if path.is_dir())


def iter_worker_dirs(factory_dir: Path) -> list[Path]:
    return sorted(path for path in factory_dir.iterdir() if path.is_dir())


def iter_annotation_paths(worker_dir: Path) -> list[Path]:
    return sorted(path for path in worker_dir.glob("*.txt") if path.is_file())


def output_csv_path(output_root: Path, factory: str, worker: str, annotation_path: Path) -> Path:
    return output_root / factory / worker / f"{annotation_path.stem}.csv"


def render_prompt(prompt_template: str, factory: str, worker: str) -> str:
    return prompt_template.replace("{factory}", factory).replace("{worker}", worker)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_model_events(response_text: str) -> list[dict]:
    candidates: list[str] = []
    stripped = strip_code_fences(response_text)
    candidates.append(stripped)

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(stripped[first_bracket : last_bracket + 1])

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            return data["events"]

    raise ValueError(f"Model response was not a valid JSON event list: {response_text[:500]}")


def extract_section(report_text: str, heading: str) -> str:
    if heading not in REPORT_HEADINGS:
        raise ValueError(f"Unknown report heading: {heading}")

    heading_index = REPORT_HEADINGS.index(heading)
    match = re.search(rf"(?m)^{re.escape(heading)}\s*$", report_text)
    if not match:
        raise ValueError(f"Could not find heading '{heading}' in factory report.")

    start = match.end()
    end = len(report_text)
    for next_heading in REPORT_HEADINGS[heading_index + 1 :]:
        next_match = re.search(rf"(?m)^{re.escape(next_heading)}\s*$", report_text[start:])
        if next_match:
            end = start + next_match.start()
            break
    return report_text[start:end].strip()


def parse_bullet_list(section_text: str) -> list[str]:
    items: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("* "):
            items.append(stripped[2:].strip())
        elif stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def parse_worker_summaries(section_text: str) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("worker_") or ":" not in stripped:
            continue
        worker, summary = stripped.split(":", 1)
        summaries[worker.strip()] = summary.strip()
    return summaries


def load_factory_vocabulary(factory_report_root: Path, factory: str) -> dict:
    report_path = factory_report_root / f"{factory}.txt"
    report_text = read_text_file(report_path, "Factory report file")

    process_labels = parse_bullet_list(extract_section(report_text, "Factory Process Labels"))
    activity_labels = parse_bullet_list(extract_section(report_text, "Factory Activity Catalogue"))
    worker_summaries = parse_worker_summaries(extract_section(report_text, "Worker Process Summaries"))

    if not process_labels:
        raise ValueError(f"No factory process labels found in {report_path}")
    if not activity_labels:
        raise ValueError(f"No factory activity labels found in {report_path}")

    return {
        "process_labels": process_labels,
        "activity_labels": activity_labels,
        "worker_summaries": worker_summaries,
    }


def event_list_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "estimated_start_time": {
                            "type": "string",
                            "pattern": r"^\d{2}:\d{2}:\d{2}$",
                        },
                        "estimated_end_time": {
                            "type": "string",
                            "pattern": r"^\d{2}:\d{2}:\d{2}$",
                        },
                        "activity": {
                            "type": "string",
                        },
                        "process": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "estimated_start_time",
                        "estimated_end_time",
                        "activity",
                        "process",
                    ],
                },
            },
        },
        "required": ["events"],
    }


def generate_event_json(
    session: requests.Session,
    api_key: str,
    model: str,
    prompt: str,
    context_text: str,
) -> str:
    response_payload = request_with_retry(
        session=session,
        method="POST",
        url=f"{BASE_URL}/responses",
        api_key=api_key,
        timeout_seconds=GENERATE_READ_TIMEOUT_SECONDS,
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "process_mining_event_list",
                    "description": "A structured event list for process mining.",
                    "strict": True,
                    "schema": event_list_json_schema(),
                }
            },
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": context_text,
                        }
                    ],
                },
            ]
        },
    )

    text_parts: list[str] = []
    for output_item in response_payload.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if text:
                    text_parts.append(text)

    response_text = "\n".join(text_parts).strip()
    if response_text:
        return response_text

    raise RuntimeError(f"OpenAI returned no event JSON text.\nResponse payload: {response_payload}")


def build_context_text(
    factory: str,
    worker: str,
    process_labels: list[str],
    activity_labels: list[str],
    worker_summary: str,
    annotation_text: str,
) -> str:
    process_block = "\n".join(f"- {label}" for label in process_labels)
    activity_block = "\n".join(f"- {label}" for label in activity_labels)
    summary_block = worker_summary if worker_summary else "(not available)"

    return (
        f"Factory: {factory}\n"
        f"Worker: {worker}\n\n"
        "Allowed factory process labels:\n"
        f"{process_block}\n\n"
        "Allowed factory activity labels:\n"
        f"{activity_block}\n\n"
        "Worker summary:\n"
        f"{summary_block}\n\n"
        "Raw annotation transcript:\n"
        f"{annotation_text}"
    )


def canonicalize_label(value: str, allowed_labels: list[str], label_type: str) -> str:
    normalized_allowed = {label.strip().casefold(): label for label in allowed_labels}
    cleaned = value.strip().strip("\"'`").strip()
    canonical = normalized_allowed.get(cleaned.casefold())
    if canonical is None:
        raise ValueError(f"{label_type} label not in allowed factory list: {value}")
    return canonical


def parse_time_offset(value: object) -> timedelta:
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))

    if not isinstance(value, str):
        raise ValueError(f"Time value must be a string or number, got {type(value).__name__}")

    cleaned = value.strip()
    parts = cleaned.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported time format: {value}")

    try:
        hours_i = int(hours)
        minutes_i = int(minutes)
        seconds_i = int(seconds)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric time format: {value}") from exc

    if minutes_i < 0 or minutes_i >= 60 or seconds_i < 0 or seconds_i >= 60 or hours_i < 0:
        raise ValueError(f"Out-of-range time value: {value}")
    return timedelta(hours=hours_i, minutes=minutes_i, seconds=seconds_i)


def get_event_value(event: dict, primary_key: str, fallback_keys: tuple[str, ...] = ()) -> object:
    if primary_key in event:
        return event[primary_key]
    for fallback_key in fallback_keys:
        if fallback_key in event:
            return event[fallback_key]
    raise ValueError(f"Event missing required field '{primary_key}': {event}")


def normalize_events(
    events: list[dict],
    factory: str,
    worker: str,
    process_labels: list[str],
    activity_labels: list[str],
) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"Event entry is not an object: {event}")

        start_value = get_event_value(event, "estimated_start_time", ("start_time",))
        end_value = get_event_value(event, "estimated_end_time", ("end_time",))
        activity_value = get_event_value(event, "activity")
        process_value = get_event_value(event, "process")

        start_offset = parse_time_offset(start_value)
        end_offset = parse_time_offset(end_value)
        if end_offset < start_offset:
            raise ValueError(f"Event end time precedes start time: {event}")

        activity = canonicalize_label(str(activity_value), activity_labels, "Activity")
        process = canonicalize_label(str(process_value), process_labels, "Process")

        normalized_rows.append(
            {
                "start_timestamp": (BASE_EVENT_DATETIME + start_offset).strftime("%Y-%m-%d %H:%M:%S"),
                "end_timestamp": (BASE_EVENT_DATETIME + end_offset).strftime("%Y-%m-%d %H:%M:%S"),
                "activity": activity,
                "process": process,
                "factory": factory,
                "worker": worker,
            }
        )

    normalized_rows.sort(
        key=lambda row: (
            row["start_timestamp"],
            row["end_timestamp"],
            row["activity"],
            row["process"],
        )
    )
    return normalized_rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_annotation(
    api_key: str,
    model: str,
    prompt_template: str,
    factory: str,
    worker: str,
    annotation_path: Path,
    factory_vocabulary: dict,
    output_root: Path,
    overwrite: bool,
) -> None:
    output_path = output_csv_path(output_root, factory, worker, annotation_path)
    if output_path.exists() and not overwrite:
        print(f"  keeping existing CSV {output_path}", flush=True)
        return

    annotation_text = read_text_file(annotation_path, "Raw annotation file")
    prompt = render_prompt(prompt_template, factory=factory, worker=worker)
    worker_summary = factory_vocabulary["worker_summaries"].get(worker, "")
    context_text = build_context_text(
        factory=factory,
        worker=worker,
        process_labels=factory_vocabulary["process_labels"],
        activity_labels=factory_vocabulary["activity_labels"],
        worker_summary=worker_summary,
        annotation_text=annotation_text,
    )

    print(f"  converting {annotation_path.name} -> {output_path}", flush=True)
    with requests.Session() as session:
        response_text = generate_event_json(
            session=session,
            api_key=api_key,
            model=model,
            prompt=prompt,
            context_text=context_text,
        )
    events = parse_model_events(response_text)
    rows = normalize_events(
        events=events,
        factory=factory,
        worker=worker,
        process_labels=factory_vocabulary["process_labels"],
        activity_labels=factory_vocabulary["activity_labels"],
    )
    write_csv(output_path, rows)


def iter_targets(annotation_root: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for factory_dir in iter_factory_dirs(annotation_root):
        for worker_dir in iter_worker_dirs(factory_dir):
            targets.append((factory_dir.name, worker_dir.name))
    return targets


def collect_annotation_paths(annotation_root: Path, factory: str, worker: str) -> list[Path]:
    worker_dir = annotation_root / factory / worker
    if not worker_dir.is_dir():
        raise FileNotFoundError(f"Worker annotation directory not found: {worker_dir}")

    annotation_paths = iter_annotation_paths(worker_dir)
    if not annotation_paths:
        raise FileNotFoundError(f"No raw annotation files found under {worker_dir}")
    return annotation_paths


def build_annotation_tasks(
    annotation_root: Path,
    targets: list[tuple[str, str]],
) -> list[tuple[str, str, Path]]:
    tasks: list[tuple[str, str, Path]] = []
    for factory, worker in targets:
        annotation_paths = collect_annotation_paths(annotation_root, factory, worker)
        print(f"Worker {factory}/{worker}: {len(annotation_paths)} annotation file(s)", flush=True)
        for annotation_path in annotation_paths:
            tasks.append((factory, worker, annotation_path))
    return tasks


def preload_factory_vocabularies(
    factory_report_root: Path,
    factories: set[str],
) -> dict[str, dict]:
    vocabulary_cache: dict[str, dict] = {}
    for factory in sorted(factories):
        vocabulary_cache[factory] = load_factory_vocabulary(factory_report_root, factory)
    return vocabulary_cache


def process_annotation_task(
    api_key: str,
    model: str,
    prompt_template: str,
    output_root: Path,
    overwrite: bool,
    factory: str,
    worker: str,
    annotation_path: Path,
    factory_vocabulary: dict,
) -> tuple[str, str, str]:
    process_annotation(
        api_key=api_key,
        model=model,
        prompt_template=prompt_template,
        factory=factory,
        worker=worker,
        annotation_path=annotation_path,
        factory_vocabulary=factory_vocabulary,
        output_root=output_root,
        overwrite=overwrite,
    )
    return factory, worker, annotation_path.name


def main() -> int:
    args = parse_args()
    api_key = read_openai_api_key()
    prompt_template = read_text_file(args.prompt_file, "Prompt file")

    if args.all:
        targets = iter_targets(args.annotation_root)
        if not targets:
            print(f"No factory/worker folders found under {args.annotation_root}", flush=True)
            return 0
        print(f"Processing all workers under {args.annotation_root}", flush=True)
    else:
        targets = [(args.factory, args.worker)]

    print(f"Annotation root: {args.annotation_root}", flush=True)
    print(f"Factory report root: {args.factory_report_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}", flush=True)

    annotation_tasks = build_annotation_tasks(args.annotation_root, targets)
    if not annotation_tasks:
        print("No annotation files found to process.", flush=True)
        return 0

    vocabulary_cache = preload_factory_vocabularies(
        args.factory_report_root,
        {factory for factory, _, _ in annotation_tasks},
    )

    failed_tasks: list[tuple[str, str, str, str]] = []
    max_workers = min(MAX_CONCURRENT_TASKS, len(annotation_tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_annotation_task,
                api_key,
                args.model,
                prompt_template,
                args.output_root,
                args.overwrite,
                factory,
                worker,
                annotation_path,
                vocabulary_cache[factory],
            ): (factory, worker, annotation_path)
            for factory, worker, annotation_path in annotation_tasks
        }

        for future in as_completed(future_to_task):
            factory, worker, annotation_path = future_to_task[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failed_tasks.append((factory, worker, annotation_path.name, str(exc)))
                print(
                    f"Failed to convert {factory}/{worker}/{annotation_path.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    if failed_tasks:
        print("Finished with failures:", file=sys.stderr, flush=True)
        for factory, worker, annotation_name, message in failed_tasks:
            print(
                f"  {factory}/{worker}/{annotation_name}: {message}",
                file=sys.stderr,
                flush=True,
            )
        return 1

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
