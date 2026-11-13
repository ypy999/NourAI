import json
import pytest

from corpus.acquisition import AcquisitionError, download_resumable, sha256_file, verify_download_provenance
from data.download_manager import DownloadManager


def test_verified_download_is_reused_without_network(tmp_path,monkeypatch):
    import urllib.request
    path=tmp_path/"source.tar.bz2"; path.write_bytes(b"known source bytes")
    meta={"source_id":"tatoeba_cc0","source_url":"https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
          "license_id":"CC0-1.0","bytes":path.stat().st_size,"sha256":sha256_file(path)}
    path.with_name(path.name+".provenance.json").write_text(json.dumps(meta),encoding="utf-8")
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**k:pytest.fail("network should not be used for an existing verified file"))
    reused=download_resumable("tatoeba_cc0",path)
    assert reused["already_present"] is True
    assert verify_download_provenance("tatoeba_cc0",path)["sha256"]==meta["sha256"]


def test_tampered_download_and_nonexplicit_download_are_rejected(tmp_path):
    path=tmp_path/"source.tar.bz2"; path.write_bytes(b"data")
    meta={"source_id":"tatoeba_cc0","source_url":"https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
          "license_id":"CC0-1.0","bytes":4,"sha256":"0"*64}
    path.with_name(path.name+".provenance.json").write_text(json.dumps(meta),encoding="utf-8")
    with pytest.raises(AcquisitionError,match="SHA-256"):
        verify_download_provenance("tatoeba_cc0",path)
    with pytest.raises(PermissionError,match="explicit"):
        DownloadManager(tmp_path).download("tatoeba_cc0")


def test_download_manager_rejects_custom_paths_outside_project(tmp_path):
    config = {
        "metadata_database": "data/metadata/corpus.sqlite3",
        "raw_directory": "../outside/raw",
    }
    with pytest.raises(ValueError, match="raw directory path must remain under the project root"):
        DownloadManager(tmp_path, config=config)

    config["raw_directory"] = "data/raw/external"
    config["metadata_database"] = str(tmp_path.parent / "outside.sqlite3")
    with pytest.raises(ValueError, match="metadata database path must remain under the project root"):
        DownloadManager(tmp_path, config=config)
