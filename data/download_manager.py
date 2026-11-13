"""Source-aware download facade."""
from pathlib import Path

from corpus.acquisition import download_resumable
from corpus.licensing import require_approved_source
from corpus.pipeline import load_phase8_config
from .license_manager import LicenseManager
from .storage_manager import CorpusMetadataStore


class DownloadManager:
    def __init__(self, project_root, config=None):
        self.root = Path(project_root).resolve()
        self.config = config or load_phase8_config()
        self._metadata_database = self._resolve_under_root(self.config["metadata_database"], "metadata database")
        self._raw_directory = self._resolve_under_root(self.config["raw_directory"], "raw directory")

    def _resolve_under_root(self, path, label):
        candidate = Path(path)
        candidate = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"{label} path must remain under the project root") from exc
        return candidate

    def download(self, source_id, *, explicit_request=False, expected_sha256=None):
        if not explicit_request:
            raise PermissionError("download requires an explicit caller request")
        source = require_approved_source(source_id)
        licensing = LicenseManager(self._metadata_database)
        try:
            licensing.seed_catalog()
            source = licensing.ensure_approved(source_id)
        finally:
            licensing.close()
        path = self._raw_directory / f"{source_id}.tar.bz2"
        metadata = download_resumable(source_id, path, expected_sha256=expected_sha256,
            max_bytes=self.config["max_download_bytes"], timeout=self.config["download"]["timeout_seconds"],
            retries=self.config["download"]["retries"])
        store = CorpusMetadataStore(self.root, self._metadata_database)
        store.register_source({"source_id":source_id,**source})
        store.record_operation("download", "COMPLETE", source_id=source_id, details=metadata)
        store.close()
        return metadata
