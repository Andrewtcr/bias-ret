"""Verify every embeddings-bundle file against scripts/embeddings_manifest.csv.

Fail loud: any missing file, size mismatch, or sha256 mismatch exits non-zero
with a per-file error report.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().with_name("embeddings_manifest.csv")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MANIFEST_PATH.exists():
        sys.exit(f"[verify] missing {MANIFEST_PATH} — restore the manifest from the repository")

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))

    print(f"[verify] checking {len(rows)} bundle files")

    errors: list[str] = []
    for i, row in enumerate(rows, 1):
        path = REPO_ROOT / row["path"]
        if not path.exists():
            errors.append(f"MISSING {row['path']}")
            continue
        size = path.stat().st_size
        if size != int(row["size_bytes"]):
            errors.append(f"SIZE   {row['path']}  expected={row['size_bytes']} actual={size}")
            continue
        sha = _sha256(path)
        if sha != row["sha256"]:
            errors.append(f"SHA256 {row['path']}  expected={row['sha256'][:12]}... actual={sha[:12]}...")
            continue
        if i % 25 == 0 or i == len(rows):
            print(f"  checked {i}/{len(rows)}")

    if errors:
        print(f"\n[verify] FAILED — {len(errors)} issues:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[verify] OK — all {len(rows)} entries pass")


if __name__ == "__main__":
    main()
