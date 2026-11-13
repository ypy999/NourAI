"""Coordinator around Phase 8's approved, resumable single-source importer."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from corpus.acquisition import sha256_file
from corpus.pipeline import import_source_file
from data.source_registry import SourceRegistry
from corpus.licensing import require_approved_source


@dataclass(frozen=True)
class ImportRequest:
    source_id: str
    source_path: str | Path | None = None
    languages: tuple[str, ...] | None = None
    resume: bool = False
    download: bool = False


class MultiSourceImportCoordinator:
    """Run independently auditable imports without merging license domains.

    Each source retains its own output and report. A failed source is recorded
    and does not roll back successful imports. Import jobs require the existing
    Phase 8 approval gate; this class never grants approval.
    """

    def __init__(self, project_root, *, registry=None, importer=None, source_validator=None,
                 downloader=None, report_directory="reports/phase10/source_imports"):
        self.root = Path(project_root).resolve()
        self.registry = registry or SourceRegistry()
        self.importer = importer or import_source_file
        self.source_validator = source_validator or require_approved_source
        if downloader is None:
            from data.download_manager import DownloadManager
            downloader = DownloadManager(self.root).download
        self.downloader = downloader
        self.raw_root = (self.root / "data/raw/external").resolve()
        self.output_root = (self.root / "data/processed/phase10/imported").resolve()
        self.report_root = (self.root / report_directory).resolve()
        for candidate in (self.raw_root, self.output_root, self.report_root):
            try:
                candidate.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("Phase 10 paths must remain inside the project root") from exc

    def _source_paths(self, path):
        path = Path(path).resolve()
        try:
            path.relative_to(self.raw_root)
        except ValueError as exc:
            raise ValueError(f"source must be staged under {self.raw_root}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _output_path(self, request, digest):
        languages = sorted(set(request.languages or ()))
        suffix = hashlib.sha256(json.dumps(languages, separators=(",", ":")).encode()).hexdigest()[:8]
        name = f"{request.source_id}_{digest[:16]}_{suffix}.jsonl"
        path = self.output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_source_report(self, row):
        self.report_root.mkdir(parents=True, exist_ok=True)
        source_id = row.get("source_id", "unknown").replace("/", "_").replace("\\", "_")
        path = self.report_root / f"{source_id}_{row.get('source_sha256','unknown')[:16]}.json"
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def run(self, requests):
        """Process requests in order; source failures remain isolated and logged."""
        results = []
        seen = set()
        for request in requests:
            if not isinstance(request, ImportRequest):
                request = ImportRequest(**request)
            base = {"source_id": request.source_id, "source_path": str(request.source_path) if request.source_path else None,
                    "languages": sorted(set(request.languages or ())), "status": "FAILED"}
            try:
                source = self.source_validator(request.source_id)
                catalog = self.registry.get(request.source_id)
                if source.get("status", source.get("state")) not in ("VERIFIED",):
                    raise PermissionError(f"source {request.source_id} is not license-verified")
                if not source.get("approved_for_ingestion", False):
                    raise PermissionError(f"source {request.source_id} is not approved for ingestion")
                if request.source_path is None:
                    if not request.download:
                        raise ValueError("request source_path or download=True is required")
                    acquisition = self.downloader(request.source_id, explicit_request=True)
                    local_path = self._source_paths(acquisition["downloaded_file"])
                    base["acquisition"] = acquisition
                else:
                    local_path = self._source_paths(request.source_path)
                digest = sha256_file(local_path)
                key = (request.source_id, digest, tuple(base["languages"]))
                base.update({"source_sha256": digest, "license_id": source.get("license_id"),
                             "license_status": source.get("status", source.get("state")),
                             "approved_for_ingestion": True})
                if key in seen:
                    base.update({"status": "SKIPPED_DUPLICATE_REQUEST", "reason": "same source, file hash, and language filter already processed in this batch"})
                else:
                    seen.add(key)
                    output_path = self._output_path(request, digest)
                    imported = self.importer(request.source_id, local_path, output_path,
                                             project_root=self.root,
                                             languages=request.languages,
                                             resume=request.resume)
                    base.update({"status": "IMPORTED", "output_path": str(output_path),
                                 "records": imported.get("accepted_records"),
                                 "already_present": imported.get("already_present", False),
                                 "import_report": imported})
                base["source_name"] = catalog.get("name")
            except Exception as exc:
                base["error_type"] = type(exc).__name__
                base["error"] = str(exc)
            try:
                base["report_path"] = str(self._write_source_report(base))
            except OSError as exc:
                base["report_write_error"] = f"{type(exc).__name__}: {exc}"
            results.append(base)
        return {"schema_version": 1, "sources_requested": len(results),
                "sources_imported": sum(row["status"] == "IMPORTED" for row in results),
                "sources_failed": sum(row["status"] == "FAILED" for row in results),
                "sources_skipped": sum(row["status"] == "SKIPPED_DUPLICATE_REQUEST" for row in results),
                "results": results}
