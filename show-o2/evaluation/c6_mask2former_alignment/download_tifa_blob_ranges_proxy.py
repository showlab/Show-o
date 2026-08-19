#!/usr/bin/env python3
"""Proxy-tolerant entry point for the pinned, ranged TIFA downloader."""

from __future__ import annotations

import time
import urllib.request

from evaluation.c6_mask2former_alignment import download_tifa_blob_ranges as base


def fetch_range(start: int, end: int) -> bytes:
    expected = end - start + 1
    expected_ranges = {
        f"bytes {start}-{end}/{base.TOTAL_BYTES}",
        f"bytes {start}-{end}/*",
    }
    last_error: Exception | None = None
    for attempt in range(1, 6):
        request = urllib.request.Request(
            base.URL,
            headers={
                "Accept": "application/vnd.github.raw+json",
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "showo2-tifa-self-cycle",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or content_range not in expected_ranges:
                    raise RuntimeError(
                        f"unexpected range response status={response.status} "
                        f"content-range={content_range!r}"
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


if __name__ == "__main__":
    base.fetch_range = fetch_range
    raise SystemExit(base.main())
