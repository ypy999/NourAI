import json

from corpus.reporting import build_report, write_dashboard, write_report
from data.corpus_dashboard import build_corpus_dashboard
from data.raw_storage import make_raw_record


def test_dashboard_reports_real_file_metrics_and_license_distribution(tmp_path):
    path=tmp_path/"raw.jsonl"
    rows=[]
    for text in ("hello world","hello world","مرحبا"):
        row=make_raw_record(text,source_id="tatoeba_cc0",license_id="CC0-1.0"); row["text"]=text
        rows.append(row)
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    report=build_corpus_dashboard(path,source_statuses={"tatoeba_cc0":"VERIFIED"},duplicate_records=None)
    assert report["record_count"]==3 and report["unique_records"]==2 and report["duplicate_records"]==1
    assert report["license_distribution"]=={"CC0-1.0":3}
    assert report["quality_score_distribution"]
    assert report["processed_bytes_on_disk"]==path.stat().st_size


def test_report_pair_and_dashboard_include_training_scope(tmp_path):
    report=build_report(environment={},sources=[],downloads=[],imports=[],splits={},validation={},quality={},performance={},
                        readiness={"status":"CONDITIONALLY_READY","blockers":[],"training_readiness":{"status":"NOT_READY","blockers":["no Arabic validation"]}},
                        tests={"passed":1},limitations=["synthetic fixture"])
    paths=write_report(report,tmp_path/"report.json")
    assert "Actual model-training decision" in open(paths["markdown"],encoding="utf-8").read()
    dashboard=write_dashboard(report,tmp_path/"dashboard.html")
    assert "NOT_READY" in open(dashboard,encoding="utf-8").read()
