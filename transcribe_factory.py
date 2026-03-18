#!/usr/bin/env python3
"""Sequentially annotate one factory from Egocentric-10K with Gemini REST calls."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import tarfile
import time
from pathlib import Path
from typing import BinaryIO

import requests

BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_DATASET_ROOT = Path("/data/Egocentric-10K")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "raw_transcriptions"
DEFAULT_API_KEY_PATH = DEFAULT_PROJECT_ROOT / "google_api_key.txt"
DEFAULT_PROMPT_PATH = DEFAULT_PROJECT_ROOT / "annotation_prompt.txt"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
POLL_INTERVAL_SECONDS = 5.0
CONNECT_TIMEOUT_SECONDS = 30
UPLOAD_READ_TIMEOUT_SECONDS = 1800
GENERATE_READ_TIMEOUT_SECONDS = 1800
MAX_VIDEO_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 10.0
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ApiRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially annotate all videos for one factory from /data/Egocentric-10K "
            "and save .txt outputs into raw_transcriptions/factory_xxx/worker_xxx."
        )
    )
    parser.add_argument(
        "factory",
        help="Factory folder name, for example: factory_001",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})",
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
        help=f"Path to the annotation prompt (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model to use (default: gemini-3.1-pro-preview).",
    )
    return parser.parse_args()


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
    for attempt in range(1, MAX_VIDEO_ATTEMPTS + 1):
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
            if should_retry_exception(exc) and attempt < MAX_VIDEO_ATTEMPTS:
                print(
                    f"  transient error on {method} {url}: {exc}\n"
                    f"  sleeping {delay_seconds:.0f}s before retry {attempt + 1}/{MAX_VIDEO_ATTEMPTS}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def unwrap_file_resource(payload: dict) -> dict:
    file_resource = payload.get("file")
    if isinstance(file_resource, dict):
        return file_resource
    return payload


def guess_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def upload_video(
    session: requests.Session,
    api_key: str,
    display_name: str,
    mime_type: str,
    num_bytes: int,
    stream: BinaryIO,
) -> dict:
    start_response = session.post(
        f"{BASE_URL}/upload/v1beta/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=(CONNECT_TIMEOUT_SECONDS, GENERATE_READ_TIMEOUT_SECONDS),
    )
    raise_for_status(start_response)

    upload_url = start_response.headers.get("X-Goog-Upload-URL") or start_response.headers.get(
        "x-goog-upload-url"
    )
    if not upload_url:
        raise RuntimeError("Upload start response did not include X-Goog-Upload-URL.")

    upload_response = session.post(
        upload_url,
        headers={
            "Content-Length": str(num_bytes),
            "Content-Type": mime_type,
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        data=stream,
        timeout=(CONNECT_TIMEOUT_SECONDS, UPLOAD_READ_TIMEOUT_SECONDS),
    )
    raise_for_status(upload_response)
    return unwrap_file_resource(upload_response.json())


def get_file(session: requests.Session, api_key: str, file_name: str) -> dict:
    payload = request_with_retry(
        session=session,
        method="GET",
        url=f"{BASE_URL}/v1beta/{file_name}",
        api_key=api_key,
        timeout_seconds=GENERATE_READ_TIMEOUT_SECONDS,
    )
    return unwrap_file_resource(payload)


def wait_for_active_file(session: requests.Session, api_key: str, file_resource: dict) -> dict:
    file_name = file_resource.get("name")
    if not file_name:
        raise RuntimeError("Uploaded file response did not contain a file name.")

    while True:
        state = file_resource.get("state")
        if state == "ACTIVE":
            return file_resource
        if state == "FAILED":
            error_payload = file_resource.get("error") or {}
            raise RuntimeError(f"Gemini file processing failed for {file_name}: {error_payload}")

        print(f"  waiting for Gemini file processing: {state or 'STATE_UNSPECIFIED'}", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)
        file_resource = get_file(session, api_key, file_name)


def delete_uploaded_file(session: requests.Session, api_key: str, file_name: str) -> None:
    request_with_retry(
        session=session,
        method="DELETE",
        url=f"{BASE_URL}/v1beta/{file_name}",
        api_key=api_key,
        timeout_seconds=GENERATE_READ_TIMEOUT_SECONDS,
    )


def generate_annotation(
    session: requests.Session,
    api_key: str,
    model: str,
    file_uri: str,
    mime_type: str,
    prompt: str,
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
                        {
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri,
                            }
                        },
                        {"text": prompt},
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

    annotation = "\n".join(text_parts).strip()
    if annotation:
        return annotation

    raise RuntimeError(
        "Gemini returned no annotation text.\n"
        f"Response payload: {json.dumps(response_payload, ensure_ascii=True)}"
    )


def iter_worker_dirs(factory_dir: Path) -> list[Path]:
    workers_dir = factory_dir / "workers"
    if not workers_dir.is_dir():
        raise FileNotFoundError(f"Factory workers directory not found: {workers_dir}")
    return sorted(path for path in workers_dir.iterdir() if path.is_dir())


def iter_tar_paths(worker_dir: Path) -> list[Path]:
    return sorted(path for path in worker_dir.glob("*.tar") if path.is_file())


def process_video_member(
    session: requests.Session,
    api_key: str,
    model: str,
    prompt: str,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    output_path: Path,
) -> None:
    mime_type = guess_mime_type(member.name)
    retry_delay_seconds = INITIAL_RETRY_DELAY_SECONDS

    for attempt in range(1, MAX_VIDEO_ATTEMPTS + 1):
        extracted_file = archive.extractfile(member)
        if extracted_file is None:
            raise RuntimeError(f"Could not extract video member from archive: {member.name}")

        uploaded_file_name = None
        try:
            if attempt > 1:
                print(
                    f"  retrying {member.name} (attempt {attempt}/{MAX_VIDEO_ATTEMPTS})",
                    flush=True,
                )

            uploaded = upload_video(
                session=session,
                api_key=api_key,
                display_name=Path(member.name).name,
                mime_type=mime_type,
                num_bytes=member.size,
                stream=extracted_file,
            )
            uploaded_file_name = uploaded.get("name")
            active_file = wait_for_active_file(session, api_key, uploaded)
            file_uri = active_file.get("uri")
            if not file_uri:
                raise RuntimeError(f"Uploaded file did not provide a URI: {active_file}")

            annotation = generate_annotation(
                session=session,
                api_key=api_key,
                model=model,
                file_uri=file_uri,
                mime_type=active_file.get("mimeType", mime_type),
                prompt=prompt,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(annotation.rstrip() + "\n", encoding="utf-8")
            return
        except Exception as exc:  # noqa: BLE001
            if should_retry_exception(exc) and attempt < MAX_VIDEO_ATTEMPTS:
                print(
                    f"  transient error for {member.name}: {exc}\n"
                    f"  sleeping {retry_delay_seconds:.0f}s before retry",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(retry_delay_seconds)
                retry_delay_seconds *= 2
                continue
            raise
        finally:
            extracted_file.close()
            if uploaded_file_name:
                try:
                    delete_uploaded_file(session, api_key, uploaded_file_name)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  warning: could not delete uploaded Gemini file {uploaded_file_name}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )


def process_factory(
    session: requests.Session,
    api_key: str,
    model: str,
    prompt: str,
    dataset_root: Path,
    output_root: Path,
    factory: str,
) -> None:
    factory_dir = dataset_root / factory
    if not factory_dir.is_dir():
        raise FileNotFoundError(f"Factory directory not found: {factory_dir}")

    worker_dirs = iter_worker_dirs(factory_dir)
    if not worker_dirs:
        print(f"No worker folders found under {factory_dir}", flush=True)
        return

    for worker_dir in worker_dirs:
        tar_paths = iter_tar_paths(worker_dir)
        if not tar_paths:
            print(f"Skipping {worker_dir}: no tar files found.", flush=True)
            continue

        local_worker_output_dir = output_root / factory / worker_dir.name
        print(f"Worker {worker_dir.name}: {len(tar_paths)} archive(s)", flush=True)

        for tar_path in tar_paths:
            print(f" Opening archive {tar_path.name}", flush=True)
            with tarfile.open(tar_path, "r") as archive:
                video_members = sorted(
                    (
                        member
                        for member in archive.getmembers()
                        if member.isfile()
                        and Path(member.name).suffix.lower() in VIDEO_EXTENSIONS
                    ),
                    key=lambda member: member.name,
                )

                if not video_members:
                    print(f"  no video files found in {tar_path.name}", flush=True)
                    continue

                for member in video_members:
                    output_path = local_worker_output_dir / f"{Path(member.name).stem}.txt"
                    if output_path.exists():
                        print(f"  keeping existing annotation {output_path}", flush=True)
                        continue

                    print(f"  annotating {member.name} -> {output_path}", flush=True)
                    try:
                        process_video_member(
                            session=session,
                            api_key=api_key,
                            model=model,
                            prompt=prompt,
                            archive=archive,
                            member=member,
                            output_path=output_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  failed to annotate {member.name}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )


def main() -> int:
    args = parse_args()
    api_key = read_text_file(args.api_key_file, "API key file")
    prompt = read_text_file(args.prompt_file, "Prompt file")

    print(f"Factory: {args.factory}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Model: {args.model}", flush=True)

    with requests.Session() as session:
        process_factory(
            session=session,
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            factory=args.factory,
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
