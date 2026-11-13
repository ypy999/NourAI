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


def md5_file(path, chunk_size=1024 * 1024):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The source-page identity may be recorded under any of these names. Only the
# registry entry's canonical `url` is accepted; missing identity is an error,
# never silently filled from the registry.
SOURCE_URL_FIELDS = ("source_url", "official_item_url", "landing_page_url")


def _verify_source_identity(metadata, source):
    present = [(field, metadata.get(field)) for field in SOURCE_URL_FIELDS if metadata.get(field) not in (None, "")]
    if not present:
        raise AcquisitionError(
            "download provenance sidecar has no source identity "
            f"(one of {', '.join(SOURCE_URL_FIELDS)} is required)"
        )
    for field, value in present:
        if value != source["url"]:
            raise AcquisitionError(f"download provenance mismatch for {field}")
    recorded_download = metadata.get("download_url")
    registry_download = source.get("download_url")
    if recorded_download and registry_download and recorded_download != registry_download:
        raise AcquisitionError("download provenance mismatch for download_url")
    return present[0][0]


def _verify_license_identity(metadata, source):
    if metadata.get("license_id") in (None, ""):
        raise AcquisitionError("download provenance sidecar is missing license_id")
    if metadata["license_id"] != source["license_id"]:
        raise AcquisitionError("download provenance mismatch for license_id")
    recorded_license_url = metadata.get("license_url")
    registry_license_url = source.get("license_url")
    if recorded_license_url and registry_license_url and recorded_license_url != registry_license_url:
        raise AcquisitionError("download provenance mismatch for license_url")


def _verify_checksums(metadata, path):
    expected_size = metadata.get("bytes")
    if expected_size != path.stat().st_size:
        raise AcquisitionError("downloaded file size differs from provenance sidecar")
    recorded_md5 = metadata.get("md5")
    if recorded_md5:
        actual_md5 = md5_file(path)
        if recorded_md5.casefold() != actual_md5:
            raise AcquisitionError("downloaded file MD5 differs from provenance sidecar")
        official_md5 = metadata.get("expected_md5_from_official_metadata")
        if official_md5 and official_md5.casefold() != recorded_md5.casefold():
            raise AcquisitionError("recorded MD5 differs from the official source MD5")
    if not metadata.get("sha256"):
        raise AcquisitionError("download provenance sidecar is missing sha256")
    if metadata["sha256"] != sha256_file(path):
        raise AcquisitionError("downloaded file SHA-256 differs from provenance sidecar")


def verify_download_provenance(source_id, path):
    """Verify a download sidecar against the approved catalog entry.

    The sidecar must state its own source identity and license; neither is
    defaulted from the registry. Accepted source-identity aliases exist only
    because different acquisition tools used different field names.
    """
    source = require_approved_source(source_id)
    path = Path(path)
    sidecar = path.with_name(path.name + ".provenance.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"missing or invalid download provenance sidecar: {sidecar}") from exc
    if not isinstance(metadata, dict):
        raise AcquisitionError("download provenance sidecar root must be an object")
    if metadata.get("source_id") != source_id:
        raise AcquisitionError("download provenance mismatch for source_id")
    identity_field = _verify_source_identity(metadata, source)
    _verify_license_identity(metadata, source)
    _verify_checksums(metadata, path)
    return {**metadata, "verified_source_url_field": identity_field}


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
