"""Verify or restore the three exact agency inputs; never replace mismatched files."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
NAMES = {"tl_2020_44_tabblock20.zip", "egrid2023_data_rev2.xlsx", "county_adjacency2023.txt"}


def manifest():
    """Read the pinned names, HTTPS locations, sizes and hashes, without discovery."""
    rows = json.loads((HERE.parent/"data/input-sources.json").read_text())["files"]
    if {r["name"] for r in rows} != NAMES or len(rows) != len(NAMES):
        raise ValueError("Expected exactly the three fixed input versions")
    for row in rows:
        url = urllib.parse.urlparse(row["url"])
        if url.scheme != "https" or url.hostname not in {"www2.census.gov", "www.epa.gov"}:
            raise ValueError("Input URL is not the pinned agency HTTPS route")
        if not re.fullmatch("[a-f0-9]{64}", row["sha256"]) or not 0 < row["bytes"] < 60_000_000:
            raise ValueError("Invalid pinned hash or size")
    return rows


def consume(stream, row, output=None):
    """Hash bounded streamed bytes; optionally write a temporary file."""
    h, size = hashlib.sha256(), 0
    while True:
        chunk = stream.read(1024*1024)
        if not chunk:
            break
        size += len(chunk)
        if size > row["bytes"]:
            raise ValueError("Input exceeds pinned size: "+row["name"])
        h.update(chunk)
        if output is not None:
            output.write(chunk)
    if size != row["bytes"] or h.hexdigest() != row["sha256"]:
        raise ValueError("Exact-version mismatch: "+row["name"])
    return {"name": row["name"], "bytes": size, "sha256": h.hexdigest()}


def acquire(row, inputs, mode="verify", opener=urllib.request.urlopen):
    """Verify locally, download only absent files, or probe remote bytes without saving."""
    target = inputs/row["name"]
    if target.is_symlink():
        raise ValueError("Refuse an input symlink")
    if mode != "probe" and target.exists():
        with target.open("rb") as stream:
            result = consume(stream, row)
        return {**result, "status": "existing exact file; unchanged"}
    if mode == "verify":
        raise FileNotFoundError("Missing pinned input: "+row["name"])
    request = urllib.request.Request(row["url"], headers={"User-Agent": "Pinned-input-verification/1.0"})
    with opener(request, timeout=30) as response:
        final = response.geturl()
        parsed = urllib.parse.urlparse(final)
        if parsed.scheme != "https" or parsed.hostname not in {"www2.census.gov", "www.epa.gov"}:
            raise ValueError("Unexpected redirect outside the agency HTTPS routes")
        if mode == "probe":
            return {**consume(response, row), "status": "remote exact bytes verified; not saved",
                    "url": row["url"], "final_url": final}
        inputs.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=inputs, prefix=".acquire-", suffix=".part",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                result = consume(response, row, stream)
            # An atomic hard link fails if another process created the target.
            # No existing file, including a mismatched version, can be replaced.
            os.link(temporary, target)
            return {**result, "status": "downloaded exact file", "url": row["url"], "final_url": final}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
