import hashlib
import json

import pytest

from corpus.acquisition import AcquisitionError, md5_file, sha256_file, verify_download_provenance
from corpus.licensing import get_source
from corpus.validation import validate_jsonl


def _download(tmp_path, source_id, *, filename="source.bin", overrides=None, drop=()):
    path = tmp_path / filename
    path.write_bytes(b"provenance fixture bytes for " + source_id.encode())
    source = get_source(source_id)
    meta = {
        "source_id": source_id,
        "source_url": source["url"],
        "download_url": source.get("download_url"),
        "license_id": source["license_id"],
        "license_url": source.get("license_url"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    meta.update(overrides or {})
    for key in drop:
        meta.pop(key, None)
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_qnl_sidecar_accepts_official_item_url_identity(tmp_path):
    qnl = get_source("qnl_arabic_ocr_v2")
    path = _download(
        tmp_path,
        "qnl_arabic_ocr_v2",
        overrides={"official_item_url": qnl["url"]},
        drop=("source_url", "license_url"),
    )
    result = verify_download_provenance("qnl_arabic_ocr_v2", path)
    assert result["verified_source_url_field"] == "official_item_url"
    assert result["license_id"] == qnl["license_id"]


def test_qnl_sidecar_rejects_wrong_official_item_url(tmp_path):
    path = _download(
        tmp_path,
        "qnl_arabic_ocr_v2",
        overrides={"official_item_url": "https://example.invalid/not-qnl"},
        drop=("source_url", "license_url"),
    )
    with pytest.raises(AcquisitionError, match="official_item_url"):
        verify_download_provenance("qnl_arabic_ocr_v2", path)


def test_wto_sidecar_without_license_is_rejected_not_defaulted(tmp_path):
    path = _download(tmp_path, "wto_public_corpus_en_fr", drop=("license_id", "license_url"))
    with pytest.raises(AcquisitionError, match="missing license_id"):
        verify_download_provenance("wto_public_corpus_en_fr", path)


def test_wto_sidecar_with_registered_license_passes(tmp_path):
    source = get_source("wto_public_corpus_en_fr")
    path = _download(tmp_path, "wto_public_corpus_en_fr")
    result = verify_download_provenance("wto_public_corpus_en_fr", path)
    assert result["license_id"] == "WTO_PUBLIC_CORPUS_TERMS"
    assert result["verified_source_url_field"] == "source_url"
    assert result["download_url"] == source["download_url"]


def test_wrong_or_absent_license_and_identity_are_rejected(tmp_path):
    wrong_license = _download(tmp_path, "wto_public_corpus_en_fr", filename="wrong-license.bin",
                              overrides={"license_id": "CC0-1.0"})
    with pytest.raises(AcquisitionError, match="license_id"):
        verify_download_provenance("wto_public_corpus_en_fr", wrong_license)

    no_identity = _download(tmp_path, "wto_public_corpus_en_fr", filename="no-identity.bin",
                            drop=("source_url",))
    with pytest.raises(AcquisitionError, match="no source identity"):
        verify_download_provenance("wto_public_corpus_en_fr", no_identity)


def test_download_url_mismatch_is_rejected(tmp_path):
    path = _download(tmp_path, "wto_public_corpus_en_fr",
                     overrides={"download_url": "https://example.invalid/other.zip"})
    with pytest.raises(AcquisitionError, match="download_url"):
        verify_download_provenance("wto_public_corpus_en_fr", path)


def test_recorded_md5_must_match_file_and_official_md5(tmp_path):
    path = _download(tmp_path, "qnl_arabic_ocr_v2", filename="qnl-md5.bin",
                     overrides={"md5": "0" * 32, "expected_md5_from_official_metadata": "0" * 32})
    with pytest.raises(AcquisitionError, match="MD5"):
        verify_download_provenance("qnl_arabic_ocr_v2", path)

    path2 = _download(tmp_path, "qnl_arabic_ocr_v2", filename="qnl-md5-official.bin")
    actual = md5_file(path2)
    meta = json.loads(path2.with_name(path2.name + ".provenance.json").read_text(encoding="utf-8"))
    meta.update({"md5": actual, "expected_md5_from_official_metadata": "1" * 32})
    path2.with_name(path2.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="official source MD5"):
        verify_download_provenance("qnl_arabic_ocr_v2", path2)


def test_record_validation_uses_registry_license_not_cc0_hardcode(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"text": "نص عربي من المكتبة الوطنية", "source_id": "qnl_arabic_ocr_v2",
         "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA"},
        {"text": "An English WTO segment.", "source_id": "wto_public_corpus_en_fr",
         "license_id": "WTO_PUBLIC_CORPUS_TERMS"},
        {"text": "A French WTO segment.", "source_id": "wto_public_corpus_en_fr",
         "license_id": "WTO_PUBLIC_CORPUS_TERMS"},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    result = validate_jsonl(path, require_provenance=True)
    assert result["valid"] is True, result["errors"]
    assert result["records"] == 3
    assert result["unknown_source_records"] == 0
    assert result["unknown_license_records"] == 0


def test_record_validation_rejects_wrong_license_and_unknown_source(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [
        {"text": "QNL text with CC0 label only", "source_id": "qnl_arabic_ocr_v2", "license_id": "CC0-1.0"},
        {"text": "Unknown source text", "source_id": "some_new_crawler", "license_id": "CC0-1.0"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = validate_jsonl(path, require_provenance=True)
    assert result["valid"] is False
    assert result["unknown_license_records"] == 2
    assert result["unknown_source_records"] == 1


def test_explicit_allowlist_cannot_bypass_registered_license(tmp_path):
    path = tmp_path / "qnl.jsonl"
    path.write_text(json.dumps({"text": "نص", "source_id": "qnl_arabic_ocr_v2", "license_id": "CC0-1.0"}) + "\n",
                    encoding="utf-8")
    result = validate_jsonl(path, require_provenance=True, approved_source_ids=("qnl_arabic_ocr_v2",))
    assert result["valid"] is False
    assert result["unknown_license_records"] == 1
