"""Download the full international results dataset (1872-present).

Run this on a machine with internet access:

    python scripts/download_full_data.py

It fetches the community-maintained ``results.csv`` (martj42/international_results,
CC0) and saves it as ``data/full_international_results.csv``. Once present, the
training pipeline uses it automatically instead of the bundled World-Cup-only file,
which substantially improves Elo ratings and model calibration.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "full_international_results.csv"


def main() -> int:
    print(f"Downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            content = resp.read()
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    header = content.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    if "home_team" not in header:
        print(f"Unexpected file format (header: {header!r})", file=sys.stderr)
        return 1
    DEST.write_bytes(content)
    lines = content.count(b"\n")
    print(f"Saved {DEST} ({lines} rows). Re-run training to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
