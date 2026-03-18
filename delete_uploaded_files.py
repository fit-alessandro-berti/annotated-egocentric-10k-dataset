#!/usr/bin/env python3
"""Delete leftover uploaded Gemini files for the current API project."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "google_api_key.txt"
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300
INITIAL_RETRY_DELAY_SECONDS = 5.0
MAX_ATTEMPTS = 5
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete uploaded Gemini Files API objects owned by the current API project."
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help=f"Path to google_api_key.txt (default: {DEFAULT_API_KEY_PATH})",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only delete files whose display name starts with this prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching uploaded files without deleting them.",
    )
    return parser.parse_args()


def read_api_key(api_key_path: Path) -> str:
    if not api_key_path.is_file():
        raise FileNotFoundError(f"API key file not found: {api_key_path}")

    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {api_key_path}")
    return api_key


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
    **kwargs,
) -> dict:
    headers = dict(kwargs.pop("headers", {}))
    headers["x-goog-api-key"] = api_key

    response = session.request(
        method=method,
        url=url,
        headers=headers,
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
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
    **kwargs,
) -> dict:
    delay_seconds = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return request_json(session, method, url, api_key, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if should_retry_exception(exc) and attempt < MAX_ATTEMPTS:
                print(
                    f"transient error on {method} {url}: {exc}\n"
                    f"sleeping {delay_seconds:.0f}s before retry {attempt + 1}/{MAX_ATTEMPTS}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def iter_uploaded_files(session: requests.Session, api_key: str) -> list[dict]:
    files: list[dict] = []
    page_token = ""

    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        payload = request_with_retry(
            session=session,
            method="GET",
            url=f"{BASE_URL}/v1beta/files",
            api_key=api_key,
            params=params,
        )
        files.extend(payload.get("files", []))
        page_token = payload.get("nextPageToken", "")
        if not page_token:
            return files


def delete_file(session: requests.Session, api_key: str, file_name: str) -> None:
    request_with_retry(
        session=session,
        method="DELETE",
        url=f"{BASE_URL}/v1beta/{file_name}",
        api_key=api_key,
    )


def main() -> int:
    args = parse_args()
    api_key = read_api_key(args.api_key_file)

    with requests.Session() as session:
        files = iter_uploaded_files(session, api_key)
        matching_files = [
            file_resource
            for file_resource in files
            if (file_resource.get("displayName") or "").startswith(args.prefix)
        ]

        if not matching_files:
            if args.prefix:
                print(f"No uploaded files matched prefix {args.prefix!r}.", flush=True)
            else:
                print("No uploaded files found.", flush=True)
            return 0

        action = "Would delete" if args.dry_run else "Deleting"
        print(f"{action} {len(matching_files)} uploaded file(s).", flush=True)

        deleted_count = 0
        failed_count = 0

        for file_resource in matching_files:
            file_name = file_resource.get("name", "<missing-name>")
            display_name = file_resource.get("displayName") or "<no-display-name>"
            print(f"{action}: {file_name} ({display_name})", flush=True)

            if args.dry_run:
                continue

            try:
                delete_file(session, api_key, file_name)
                deleted_count += 1
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(f"Failed to delete {file_name}: {exc}", file=sys.stderr, flush=True)

    if args.dry_run:
        print("Dry run finished.", flush=True)
        return 0

    print(
        f"Finished. Deleted {deleted_count} file(s); {failed_count} deletion(s) failed.",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
