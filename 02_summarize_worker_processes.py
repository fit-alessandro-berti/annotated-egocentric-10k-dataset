#!/usr/bin/env python3
"""Summarize worker processes from saved Egocentric-10K transcript files."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = DEFAULT_PROJECT_ROOT / "raw_transcriptions"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "worker_process_summaries"
DEFAULT_API_KEY_PATH = DEFAULT_PROJECT_ROOT / "google_api_key.txt"
DEFAULT_PROMPT_PATH = DEFAULT_PROJECT_ROOT / "worker_process_summary_prompt.txt"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

CONNECT_TIMEOUT_SECONDS = 30
GENERATE_READ_TIMEOUT_SECONDS = 1800
MAX_REQUEST_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 10.0
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the process(es) performed by a worker from saved transcript files "
            "and write one summary per worker."
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
        help="Process all factories and all workers under the input root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Transcript root directory (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Summary output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help=f"Path to google_api_key.txt (default: {DEFAULT_API_KEY_PATH})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"Path to the summary prompt (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model to use (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing worker summaries.",
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
    headers["x-goog-api-key"] = api_key

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


def iter_factory_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root directory not found: {input_root}")
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def iter_worker_dirs(factory_dir: Path) -> list[Path]:
    return sorted(path for path in factory_dir.iterdir() if path.is_dir())


def iter_transcript_paths(worker_dir: Path) -> list[Path]:
    return sorted(path for path in worker_dir.glob("*.txt") if path.is_file())


def render_prompt(prompt_template: str, factory: str, worker: str) -> str:
    return prompt_template.replace("{factory}", factory).replace("{worker}", worker)


def build_transcript_bundle(worker_dir: Path) -> tuple[int, str]:
    transcript_paths = iter_transcript_paths(worker_dir)
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript files found under {worker_dir}")

    sections: list[str] = []
    for transcript_path in transcript_paths:
        transcript = read_text_file(transcript_path, "Transcript file")
        sections.append(f"Transcript\n{transcript}")

    return len(transcript_paths), "\n\n==========\n\n".join(sections)


def generate_summary(
    session: requests.Session,
    api_key: str,
    model: str,
    prompt: str,
    transcript_bundle: str,
) -> str:
    response_payload = request_with_retry(
        session=session,
        method="POST",
        url=f"{BASE_URL}/v1beta/models/{model}:generateContent",
        api_key=api_key,
        timeout_seconds=GENERATE_READ_TIMEOUT_SECONDS,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "text": (
                                "Below are the worker transcripts. They intentionally omit file names "
                                "and should be synthesized into one worker-level report.\n\n"
                                f"{transcript_bundle}"
                            )
                        },
                    ]
                }
            ]
        },
    )

    text_parts: list[str] = []
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                text_parts.append(text)

    summary = "\n".join(text_parts).strip()
    if summary:
        return summary

    raise RuntimeError(f"Gemini returned no summary text.\nResponse payload: {response_payload}")


def summary_output_path(output_root: Path, factory: str, worker: str) -> Path:
    return output_root / factory / f"{worker}.txt"


def process_worker(
    session: requests.Session,
    api_key: str,
    model: str,
    prompt_template: str,
    input_root: Path,
    output_root: Path,
    factory: str,
    worker: str,
    overwrite: bool,
) -> None:
    worker_dir = input_root / factory / worker
    if not worker_dir.is_dir():
        raise FileNotFoundError(f"Worker directory not found: {worker_dir}")

    output_path = summary_output_path(output_root, factory, worker)
    if output_path.exists() and not overwrite:
        print(f"Keeping existing summary {output_path}", flush=True)
        return

    transcript_count, transcript_bundle = build_transcript_bundle(worker_dir)
    prompt = render_prompt(prompt_template, factory=factory, worker=worker)

    print(
        f"Summarizing {factory}/{worker} from {transcript_count} transcript file(s) -> {output_path}",
        flush=True,
    )
    summary = generate_summary(
        session=session,
        api_key=api_key,
        model=model,
        prompt=prompt,
        transcript_bundle=transcript_bundle,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.rstrip() + "\n", encoding="utf-8")


def iter_targets(input_root: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for factory_dir in iter_factory_dirs(input_root):
        for worker_dir in iter_worker_dirs(factory_dir):
            targets.append((factory_dir.name, worker_dir.name))
    return targets


def main() -> int:
    args = parse_args()
    api_key = read_text_file(args.api_key_file, "API key file")
    prompt_template = read_text_file(args.prompt_file, "Prompt file")

    if args.all:
        targets = iter_targets(args.input_root)
        if not targets:
            print(f"No factory/worker folders found under {args.input_root}", flush=True)
            return 0
        print(f"Processing all workers under {args.input_root}", flush=True)
    else:
        targets = [(args.factory, args.worker)]

    print(f"Input root: {args.input_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Model: {args.model}", flush=True)

    failed_targets: list[tuple[str, str, str]] = []
    with requests.Session() as session:
        for factory, worker in targets:
            try:
                process_worker(
                    session=session,
                    api_key=api_key,
                    model=args.model,
                    prompt_template=prompt_template,
                    input_root=args.input_root,
                    output_root=args.output_root,
                    factory=factory,
                    worker=worker,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001
                if not args.all:
                    raise
                failed_targets.append((factory, worker, str(exc)))
                print(
                    f"Failed to summarize {factory}/{worker}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    if failed_targets:
        print("Finished with failures:", file=sys.stderr, flush=True)
        for factory, worker, message in failed_targets:
            print(f"  {factory}/{worker}: {message}", file=sys.stderr, flush=True)
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
