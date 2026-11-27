"""Download a PINNED revision of P-Bench (May2222/P-Bench, CC BY 4.0) and verify it (SPEC v3.1 Step 3.1).

Standard library only. Files go under data/pbench/<revision>/ (git-ignored). A SHA-256 manifest is
written after download. The agent never reads this directory directly: the runner passes only the
question text and the CSV to the agent; answer keys stay evaluator-side.

Running the download is a network action that requires the maintainer's approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "May2222/P-Bench"
PINNED_REVISION = "402205db5904b45b3f1f0aca895980f712fd61d3"
API = "https://huggingface.co/api/datasets/{repo}/revision/{rev}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def long_path(p: Path) -> Path:
    """Windows extended-length path (the \\\\?\\ prefix) so deep task folders work under long base paths."""
    if os.name != "nt":
        return p
    prefix = "\\\\?\\"
    s = str(p.resolve())
    return p if s.startswith(prefix) else Path(prefix + s)


def local_dir(root: Path = Path("data/pbench"), rev: str = PINNED_REVISION) -> Path:
    return long_path(root / rev[:12])


def list_files(rev: str = PINNED_REVISION) -> list[str]:
    with urllib.request.urlopen(API.format(repo=REPO, rev=rev), timeout=60) as r:  # noqa: S310 - fixed host
        info = json.loads(r.read().decode())
    if info.get("sha") != rev:
        raise RuntimeError(f"revision mismatch: asked {rev}, got {info.get('sha')}")
    return [s["rfilename"] for s in info.get("siblings", [])]


def _fetch(path: str, dest: Path, rev: str) -> int:
    out = dest / path
    if out.exists() and out.stat().st_size > 0:
        return out.stat().st_size
    out.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE.format(repo=REPO, rev=rev, path=path)
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - fixed host
        data = r.read()
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return len(data)


def download(root: Path = Path("data/pbench"), rev: str = PINNED_REVISION, workers: int = 8) -> Path:
    dest = local_dir(root, rev)
    files = [f for f in list_files(rev) if not f.startswith("assets/")]
    with ThreadPoolExecutor(workers) as ex:
        sizes = list(ex.map(lambda p: _fetch(p, dest, rev), files))
    lines = []
    for p in sorted(files):
        lines.append(f"{hashlib.sha256((dest / p).read_bytes()).hexdigest()}  {p}")
    (dest / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / "DOWNLOAD.json").write_text(
        json.dumps(
            {"repo": REPO, "revision": rev, "files": len(files), "bytes": sum(sizes), "licence": "CC BY 4.0"}, indent=2
        ),
        encoding="utf-8",
    )
    return dest


def verify(dest: Path) -> bool:
    for line in (dest / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        h, p = line.split("  ", 1)
        if hashlib.sha256((dest / p).read_bytes()).hexdigest() != h:
            return False
    return True
