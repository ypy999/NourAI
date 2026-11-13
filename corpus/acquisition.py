"""Bounded, resumable HTTP downloads with hash and provenance metadata."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .licensing import require_approved_source


class AcquisitionError(RuntimeError):
    pass


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_download_provenance(source_id, path):
    """Verify local sidecar fields and content hash against an approved catalog entry."""
    source = require_approved_source(source_id)
    path = Path(path)
    sidecar = path.with_name(path.name + ".provenance.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"missing or invalid download provenance sidecar: {sidecar}") from exc
    expected = {"source_id": source_id, "source_url": source["url"], "license_id": source["license_id"]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AcquisitionError(f"download provenance mismatch for {key}")
    if metadata.get("bytes") != path.stat().st_size:
        raise AcquisitionError("downloaded file size differs from provenance sidecar")
    digest = sha256_file(path)
    if metadata.get("sha256") != digest:
        raise AcquisitionError("downloaded file SHA-256 differs from provenance sidecar")
    return metadata


def download_resumable(source_id, destination, *, expected_sha256=None,
                       max_bytes=100 * 1024 * 1024, timeout=30, retries=3):
    """Download an approved source. Existing final files are never overwritten."""
    source = require_approved_source(source_id)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verified_existing = verify_download_provenance(source_id, destination)
        return {**verified_existing, "already_present": True, "downloaded_file": str(destination)}
    meta_path = destination.with_name(destination.name + ".provenance.json")
    if meta_path.exists():
        raise FileExistsError(f"provenance sidecar already exists: {meta_path}")
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        raise AcquisitionError("refusing to follow a symlink partial download")
    last_error = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > max_bytes:
            raise AcquisitionError("partial download exceeds configured size limit")
        headers = {"User-Agent": "NourAI-Corpus-Acquisition/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(source["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                announced = response.headers.get("Content-Length")
                if announced is not None and offset + int(announced) > max_bytes:
                    raise AcquisitionError(f"download exceeds maximum size of {max_bytes} bytes")
                count = offset
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > max_bytes:
                            raise AcquisitionError("download exceeded configured size limit")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                digest = sha256_file(partial)
                if expected_sha256 and digest.casefold() != expected_sha256.casefold():
                    raise AcquisitionError("download SHA-256 does not match expected value")
                os.replace(partial, destination)
                metadata = {
                    "source_id": source_id,
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "license_id": source["license_id"],
                    "license_url": source["license_url"],
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                    "checksum_provided_by_source": bool(expected_sha256),
                    "downloaded_file": str(destination),
                    "resumable_partial": False,
                }
                meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return metadata
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    raise AcquisitionError(f"download failed after {retries} attempts; partial file retained at {partial}: {last_error}")
