"""Public acquisition API without auto-download behavior."""
from corpus.acquisition import AcquisitionError, download_resumable, sha256_file, verify_download_provenance

__all__ = ["AcquisitionError", "download_resumable", "sha256_file", "verify_download_provenance"]
