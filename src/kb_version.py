"""
kb_version.py — Knowledge-base versioning, atomic publishing, and the
stale-index fingerprint that every consumer (dashboard, Mattermost bot, API)
uses to auto-reload after a rebuild.

Why this module exists
----------------------
The old design loaded the FAISS index + BM25 into module-level globals exactly
once per process (``vector_store._INDEX`` / the bot's ``_warm_done`` /
Streamlit's ``@st.cache_resource``). After a knowledge-base rebuild in a
*separate* process (a cron job, the "Rebuild" admin action, a CI refresh) the
long-running dashboard and bot processes kept serving the OLD index forever,
until someone manually restarted them. That is the root cause of "the dashboard
and Mattermost give different answers" and "Mattermost is stuck on a stale KB".

The fix has two halves, both here:

1. **Atomic publish.** A new index is written to temp files and ``os.replace``\\d
   into place (atomic on one filesystem), and the manifest — the "current
   version" pointer — is written *last*. A reader therefore never observes a
   half-written index, and a crashed build never replaces a good index with a
   broken one.

2. **A cheap fingerprint.** ``current.json`` records a monotonically-changing
   ``version`` string. Consumers read this tiny file on each request (cheap) and
   reload their in-memory index only when the version changes. Because every
   process reads the same file, a rebuild is detected everywhere with no
   restart.

The on-disk index format (``faiss.index`` + ``metadata.pkl`` + ``metadata.json``
in ``data/index/``) is unchanged, so nothing downstream needs to know about this
module to keep working — it just gains auto-reload and atomic safety.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src import config
from src.utils import get_logger

logger = get_logger("kb_version", config.SCRAPE_LOG)

_MANIFEST_SCHEMA = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_version() -> str:
    """A sortable, unique build id, e.g. ``20260904T130501Z-a1b2c3d4``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.urandom(4).hex()}"


def file_sha256(path: Path, _chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (empty string if missing)."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def compute_index_hash(faiss_path: Optional[Path] = None,
                       metadata_path: Optional[Path] = None) -> str:
    """
    Content fingerprint of the built index = sha256(faiss bytes) xor-joined with
    sha256(metadata bytes). Stable across processes and machines, so it doubles
    as an integrity check and a dedup key between builds.
    """
    faiss_path = faiss_path or config.FAISS_INDEX
    metadata_path = metadata_path or config.METADATA_FILE
    combined = f"{file_sha256(faiss_path)}:{file_sha256(metadata_path)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]


# ── Manifest read/write ──────────────────────────────────────────────────────

def read_manifest() -> Optional[Dict]:
    """Return the active KB manifest dict, or ``None`` if there is none yet."""
    mf = config.KB_MANIFEST_FILE
    if not mf.exists():
        return None
    try:
        with open(mf, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - corrupt manifest
        logger.warning(f"Could not read KB manifest {mf}: {exc}")
        return None


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def current_fingerprint() -> str:
    """
    A cheap value that changes iff the active KB changed. Consumers compare this
    on each request to decide whether to reload — reading it is a single small
    file stat/read, not a hash of the whole index.

    Order of preference:
      1. manifest ``version`` (the normal, versioned case);
      2. manifest file mtime (manifest present but somehow versionless);
      3. faiss.index mtime+size (legacy index built before this module existed).
    """
    mf = read_manifest()
    if mf and mf.get("version"):
        return str(mf["version"])
    if config.KB_MANIFEST_FILE.exists():
        return f"manifest-mtime:{config.KB_MANIFEST_FILE.stat().st_mtime_ns}"
    fi = config.FAISS_INDEX
    if fi.exists():
        st = fi.stat()
        return f"legacy:{st.st_mtime_ns}:{st.st_size}"
    return "none"


def current_version() -> str:
    mf = read_manifest()
    return str(mf.get("version")) if mf else "unversioned"


def build_manifest(
    *,
    version: str,
    counts: Dict[str, int],
    source_counts: Optional[Dict[str, int]] = None,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    build_duration_s: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble a manifest dict (does not write it)."""
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "version": version,
        "built_at": _now_iso(),
        "index_hash": compute_index_hash(),
        "counts": counts,                       # {documents, chunks, vectors}
        "source_counts": source_counts or {},   # {website: n, commit_kb: n, ...}
        "embedding_model": embedding_model or config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "llm_model": llm_model or config.GROQ_MODEL,
    }
    if build_duration_s is not None:
        manifest["build_duration_s"] = round(float(build_duration_s), 2)
    if extra:
        manifest.update(extra)
    return manifest


def publish(manifest: Dict, snapshot: bool = True) -> Dict:
    """
    Publish ``manifest`` as the active KB. The index files are assumed to already
    be in place at ``config.FAISS_INDEX`` / ``METADATA_FILE`` (written atomically
    by the builder). Writing the manifest last is what flips consumers over to
    the new version.

    When ``snapshot`` is True, a copy of the index + metadata is also stored
    under ``data/index/versions/<version>/`` so a later build can be rolled back
    to this one. Old snapshots beyond the retention limit are pruned.
    """
    version = manifest["version"]
    if snapshot:
        try:
            _snapshot_version(version)
            manifest["snapshot"] = True
        except Exception as exc:  # snapshotting must never block a publish
            logger.warning(f"KB snapshot for {version} failed (non-fatal): {exc}")
            manifest["snapshot"] = False

    _atomic_write_json(config.KB_MANIFEST_FILE, manifest)
    logger.info(
        "Published KB version=%s hash=%s counts=%s",
        version, manifest.get("index_hash"), manifest.get("counts"),
    )
    _prune_snapshots()
    return manifest


def _version_dir(version: str) -> Path:
    return config.INDEX_VERSIONS_DIR / version


def _snapshot_version(version: str) -> None:
    vdir = _version_dir(version)
    vdir.mkdir(parents=True, exist_ok=True)
    from src.vector_store import METADATA_JSON
    for src_path in (config.FAISS_INDEX, config.METADATA_FILE, METADATA_JSON):
        if Path(src_path).exists():
            shutil.copy2(src_path, vdir / Path(src_path).name)


def list_versions() -> List[Dict]:
    """Available snapshot versions, newest first, each with its manifest-ish info."""
    d = config.INDEX_VERSIONS_DIR
    if not d.exists():
        return []
    versions = sorted((p.name for p in d.iterdir() if p.is_dir()), reverse=True)
    active = current_version()
    return [{"version": v, "active": v == active,
             "path": str(_version_dir(v))} for v in versions]


def _prune_snapshots(keep: int = 5) -> None:
    """Keep only the newest ``keep`` snapshots plus the active one."""
    d = config.INDEX_VERSIONS_DIR
    if not d.exists():
        return
    active = current_version()
    versions = sorted((p for p in d.iterdir() if p.is_dir()),
                      key=lambda p: p.name, reverse=True)
    for old in versions[keep:]:
        if old.name == active:
            continue
        try:
            shutil.rmtree(old)
            logger.info("Pruned old KB snapshot %s", old.name)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not prune snapshot {old}: {exc}")


def rollback(version: str) -> Dict:
    """
    Restore a previously-snapshotted version as the active KB. Copies the
    snapshot's files back into place atomically and republishes the manifest.
    Raises FileNotFoundError if the snapshot is missing/incomplete.
    """
    vdir = _version_dir(version)
    from src.vector_store import METADATA_JSON
    faiss_snap = vdir / config.FAISS_INDEX.name
    meta_snap = vdir / config.METADATA_FILE.name
    if not (faiss_snap.exists() and meta_snap.exists()):
        raise FileNotFoundError(f"Snapshot for version {version!r} is missing or incomplete.")

    for snap, dest in ((faiss_snap, config.FAISS_INDEX),
                       (meta_snap, config.METADATA_FILE),
                       (vdir / METADATA_JSON.name, METADATA_JSON)):
        if snap.exists():
            fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
            os.close(fd)
            shutil.copy2(snap, tmp)
            os.replace(tmp, dest)

    from src.vector_store import index_stats
    try:
        stats = index_stats()
        counts = {"documents": stats["total_documents"], "chunks": stats["total_chunks"],
                  "vectors": stats["total_chunks"]}
    except Exception:
        counts = {}
    manifest = build_manifest(version=f"{version}-rollback-{new_version()}", counts=counts,
                              extra={"rolled_back_from": version})
    return publish(manifest, snapshot=False)
