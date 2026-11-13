from pathlib import Path

from data.multi_source import ImportRequest, MultiSourceImportCoordinator


class Registry:
    def get(self, source_id):
        return {"source_id": source_id, "name": source_id.upper(), "license_id": "CC0-1.0",
                "url": "https://example.invalid/source"}


def test_multisource_isolated_import_failure_and_duplicate_request(tmp_path):
    root = tmp_path
    raw = root / "data/raw/external"
    raw.mkdir(parents=True)
    a = raw / "a.jsonl"; a.write_text('{"text":"alpha"}\n', encoding="utf-8")
    b = raw / "b.txt"; b.write_text("beta", encoding="utf-8")
    calls = []

    def validator(source_id):
        if source_id == "rejected":
            raise PermissionError("license is not approved")
        return {"source_id": source_id, "status": "VERIFIED", "approved_for_ingestion": True,
                "license_id": "CC0-1.0"}

    def importer(source_id, source_path, output_path, **kwargs):
        calls.append((source_id, Path(source_path).name))
        if source_id == "broken":
            raise ValueError("fixture import failure")
        return {"accepted_records": 1}

    coordinator = MultiSourceImportCoordinator(root, registry=Registry(), importer=importer,
                                                source_validator=validator)
    result = coordinator.run([
        ImportRequest("source_a", a), ImportRequest("source_b", b),
        ImportRequest("source_a", a), ImportRequest("broken", b),
        ImportRequest("rejected", a),
    ])
    assert result["sources_imported"] == 2
    assert result["sources_skipped"] == 1
    assert result["sources_failed"] == 2
    assert calls == [("source_a", "a.jsonl"), ("source_b", "b.txt"), ("broken", "b.txt")]
    assert all(Path(row["report_path"]).is_file() for row in result["results"])
    assert result["results"][-1]["error_type"] == "PermissionError"


def test_download_path_is_only_used_for_explicit_requests(tmp_path):
    raw = tmp_path / "data/raw/external"
    raw.mkdir(parents=True)
    archive = raw / "source.tar.bz2"
    archive.write_text("fixture", encoding="utf-8")
    downloads = []
    def downloader(source_id, *, explicit_request=False):
        downloads.append((source_id, explicit_request))
        return {"downloaded_file": str(archive)}
    def importer(source_id, source_path, output_path, **kwargs):
        return {"accepted_records": 1}
    coordinator = MultiSourceImportCoordinator(tmp_path, registry=Registry(), importer=importer,
                                                source_validator=lambda source_id: {"status":"VERIFIED","approved_for_ingestion":True,"license_id":"CC0-1.0"},
                                                downloader=downloader)
    result = coordinator.run([ImportRequest("source_a", download=True)])
    assert result["sources_imported"] == 1
    assert downloads == [("source_a", True)]
