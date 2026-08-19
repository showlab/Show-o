#!/usr/bin/env python3
"""Resumably download the pinned TIFA blob through GitHub byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from pathlib import Path


BLOB_SHA1 = "6a563f9c157e5b2a24b9f4936edd3934a8a70add"
TOTAL_BYTES = 9_630_428
URL = f"https://api.github.com/repos/Yushi-Hu/tifa/git/blobs/{BLOB_SHA1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    return parser.parse_args()


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def fetch_range(start: int, end: int) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, 6):
        request = urllib.request.Request(
            URL,
            headers={
                "Accept": "application/vnd.github.raw+json",
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "showo2-tifa-self-cycle",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content_range = response.headers.get("Content-Range", "")
                expected_range = f"bytes {start}-{end}/{TOTAL_BYTES}"
                if response.status != 206 or content_range != expected_range:
                    raise RuntimeError(
                        f"unexpected range response status={response.status} "
                        f"content-range={content_range!r} expected={expected_range!r}"
                    )
                content = response.read()
            if len(content) != expected:
                raise RuntimeError(
                    f"range {start}-{end} returned {len(content)} bytes, expected {expected}"
                )
            return content
        except Exception as error:
            last_error = error
            if attempt < 5:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


def main() -> int:
    args = parse_args()
    if args.chunk_bytes < 1:
        raise ValueError("chunk-bytes must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    completed = partial.stat().st_size if partial.is_file() else 0
    if completed > TOTAL_BYTES or completed % args.chunk_bytes:
        raise ValueError(
            f"partial size {completed} is not a resumable chunk boundary"
        )
    while completed < TOTAL_BYTES:
        end = min(completed + args.chunk_bytes, TOTAL_BYTES) - 1
        content = fetch_range(completed, end)
        with partial.open("ab") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        completed += len(content)
        print(
            f"downloaded={completed}/{TOTAL_BYTES} percent={100*completed/TOTAL_BYTES:.2f}",
            flush=True,
        )
    content = partial.read_bytes()
    actual = git_blob_sha1(content)
    if len(content) != TOTAL_BYTES or actual != BLOB_SHA1:
        raise RuntimeError(
            f"completed TIFA blob failed validation: bytes={len(content)} sha={actual}"
        )
    os.replace(partial, args.output)
    print(f"validated={args.output.resolve()} bytes={len(content)} git_blob_sha1={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
