#!/usr/bin/env python3
"""
package_kb.py — bundle the built knowledge base into a single artifact.

CI builds + validates the index, then runs this to produce ``kb-latest.tar.gz``
(the FAISS index, metadata, and the version manifest). That artifact is attached
to a GitHub Release so the always-on backend can fetch exactly the version CI
published — the backend never has to crawl or embed at request time.

    python scripts/package_kb.py                 # -> dist/kb-latest.tar.gz
    python scripts/package_kb.py --out /tmp/x.tar.gz
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def package(out: Path) -> Path:
    members = [config.FAISS_INDEX, config.METADATA_FILE,
               config.INDEX_DIR / "metadata.json", config.KB_MANIFEST_FILE]
    present = [m for m in members if Path(m).exists()]
    if not any(Path(config.FAISS_INDEX).exists() for _ in [0]):
        raise SystemExit("No FAISS index found — build the KB before packaging.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for m in present:
            tar.add(m, arcname=f"index/{Path(m).name}")
    print(f"Packaged {len(present)} file(s) -> {out} "
          f"({out.stat().st_size/1_048_576:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Package the built KB into a tarball.")
    ap.add_argument("--out", default="dist/kb-latest.tar.gz")
    args = ap.parse_args()
    package(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
