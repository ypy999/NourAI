"""Summary metrics for a registered dataset; no synthetic rows are included."""
from collections import Counter
import sqlite3
import tempfile
from pathlib import Path
import json

from .quality import score_record
from .multilingual import estimate_language
from .deduplication import DiskDeduplicator


def build_corpus_dashboard(records_path, *, source_statuses=None, imported_files=None, rejected_records=0,
                           duplicate_records=None, unique_records=None, download_statuses=None):
    path = Path(records_path)
    languages = Counter()
    licenses = Counter()
    sources = Counter()
    quality_total = 0.0
    quality_by_language = Counter()
    language_quality_count = Counter()
    quality_buckets = Counter()
    records = characters = 0
    exact_duplicates = normalized_duplicates = 0
    with tempfile.TemporaryDirectory(prefix="nourai-dashboard-",dir=path.parent) as temp:
        db=sqlite3.connect(str(Path(temp)/"dedup.sqlite3")); dedup=DiskDeduplicator(db)
        try:
            with path.open("rb") as stream:
                for line_number, raw in enumerate(stream,1):
                    row = json.loads(raw.decode("utf-8", errors="strict"))
                    text = row.get("raw_text",row.get("text"))
                    if not isinstance(text, str):
                        continue
                    duplicate=dedup.add_if_unique({"text":text,"source":str(path),"line_number":line_number})
                    if duplicate == "exact": exact_duplicates += 1
                    elif duplicate == "normalized": normalized_duplicates += 1
                    estimate = row.get("language_estimate") or estimate_language(text, source_language_code=row.get("source_language_code"))
                    language = estimate["language"]
                    score = score_record(text,duplicate=duplicate is not None)["score"]
                    bucket="0-49" if score<50 else "50-69" if score<70 else "70-89" if score<90 else "90-100"
                    quality_buckets[bucket] += 1
                    records += 1; characters += len(text)
                    languages[language] += 1
                    licenses[row.get("license_id", "UNKNOWN_LICENSE")] += 1
                    sources[row.get("source_id", "UNKNOWN_SOURCE")] += 1
                    quality_total += score; quality_by_language[language] += score; language_quality_count[language] += 1
        finally:
            db.close()
    duplicate_count=exact_duplicates+normalized_duplicates if duplicate_records is None else duplicate_records
    unique = records - duplicate_count if unique_records is None else unique_records
    input_total=records+rejected_records
    return {"record_count": records, "unique_records": unique, "duplicate_records": duplicate_count,
            "exact_duplicates":exact_duplicates,"normalized_duplicates":normalized_duplicates,
            "duplicate_fraction": duplicate_count / records if records else 0.0,
            "rejected_records": rejected_records, "rejected_fraction": rejected_records / input_total if input_total else 0.0,
            "language_distribution": dict(languages), "language_fractions": {k:v/records for k,v in languages.items()} if records else {},
            "source_distribution": dict(sources), "license_distribution": dict(licenses),
            "mean_quality_score": quality_total / records if records else None,
            "mean_quality_by_language": {k:quality_by_language[k]/v for k,v in language_quality_count.items()},
            "quality_score_distribution":dict(quality_buckets),
            "character_count": characters, "processed_bytes_on_disk": path.stat().st_size,
            "imported_file_count": len(imported_files or [path]), "source_statuses": source_statuses or {},
            "download_statuses":download_statuses or {},
            "dataset_state": "EMPTY" if records == 0 else "IMPORTED_REQUIRES_REVIEW"}


def write_dashboard_json(report, path):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path
