"""Explicit Phase 8 corpus commands; acquisition/import require explicit flags."""
import argparse
import json
from pathlib import Path

from .benchmark import benchmark_jsonl, benchmark_source_records
from .licensing import SOURCES
from .pipeline import import_source_file, load_phase8_config, prepare_splits
from .reporting import write_dashboard
from .storage import StorageManager
from .validation import validate_jsonl, validate_splits
from data.download_manager import DownloadManager
from data.license_manager import LicenseManager
from data.source_registry import SourceRegistry
from data.storage_manager import CorpusMetadataStore


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m corpus", description="NourAI Phase 8 corpus tools")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="list registered sources and their license status")
    licenses = commands.add_parser("licenses", help="show license status or append a manual review decision")
    licenses.add_argument("--review-source")
    licenses.add_argument("--state", choices=("VERIFIED","NEEDS_REVIEW","UNKNOWN_LICENSE","REJECTED"))
    licenses.add_argument("--reviewer")
    licenses.add_argument("--reason")
    licenses.add_argument("--evidence-url")
    licenses.add_argument("--license-id")
    licenses.add_argument("--license-snapshot")
    licenses.add_argument("--approve-for-ingestion", action="store_true")
    download = commands.add_parser("download", help="download an approved source")
    download.add_argument("source_id")
    download.add_argument("--confirm-cc0-source", action="store_true", help="explicitly confirm this source should be downloaded")
    imp = commands.add_parser("import", help="import a downloaded approved source into new JSONL")
    imp.add_argument("source_id")
    imp.add_argument("source_file")
    imp.add_argument("output_file")
    imp.add_argument("--confirm-cc0-source", action="store_true", help="explicitly confirm only the declared source/license is being imported")
    prep = commands.add_parser("prepare", help="run Phase 7 deduplication and split on imported JSONL")
    prep.add_argument("imported_files", nargs="+")
    prep.add_argument("--output-directory")
    process = commands.add_parser("process", help="normalize, validate, deduplicate and split imported data")
    process.add_argument("imported_files", nargs="+")
    process.add_argument("--output-directory")
    val = commands.add_parser("validate", help="validate one JSONL file")
    val.add_argument("path")
    val.add_argument("--require-provenance", action="store_true")
    val.add_argument("--validation")
    val.add_argument("--test")
    report = commands.add_parser("report", help="show the latest Phase 8 JSON/Markdown report")
    report.add_argument("--path", default="reports/phase8_corpus_readiness.json")
    commands.add_parser("readiness", help="show current readiness decision from the latest report")
    commands.add_parser("status", help="show local source, dataset, metadata and download status")
    dash = commands.add_parser("dashboard", help="render an offline dashboard from a report JSON")
    dash.add_argument("report_json")
    dash.add_argument("output_html")
    bench = commands.add_parser("benchmark", help="benchmark JSONL streaming on an existing file")
    bench.add_argument("path")
    bench_source = commands.add_parser("benchmark-source", help="benchmark an acquired source archive")
    bench_source.add_argument("path")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "sources":
        print(json.dumps(SourceRegistry().list_sources(), ensure_ascii=False, indent=2))
    elif args.command == "licenses":
        config=load_phase8_config()
        manager=LicenseManager(root/config["metadata_database"])
        try:
            manager.seed_catalog(SourceRegistry())
            if args.review_source:
                if not all((args.state,args.reviewer,args.reason)):
                    parser.error("manual review requires --state, --reviewer, and --reason")
                result=manager.review(args.review_source,args.state,reviewer=args.reviewer,rationale=args.reason,
                    evidence_url=args.evidence_url,license_snapshot=args.license_snapshot,license_id=args.license_id,
                    approve_for_ingestion=args.approve_for_ingestion)
            else:
                result={"licenses":manager.list_current(),"history":{item["source_id"]:manager.history(item["source_id"]) for item in manager.list_current()}}
            print(json.dumps(result,ensure_ascii=False,indent=2))
        finally:
            manager.close()
    elif args.command == "download":
        if not args.confirm_cc0_source:
            parser.error("download requires --confirm-cc0-source")
        result = DownloadManager(root).download(args.source_id, explicit_request=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "import":
        if not args.confirm_cc0_source:
            parser.error("import requires --confirm-cc0-source")
        result = import_source_file(args.source_id, root / args.source_file, args.output_file, project_root=root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command in ("prepare","process"):
        config = load_phase8_config()
        output = args.output_directory or config["splits_directory"]
        config = dict(config, splits_directory=output)
        result = prepare_splits([root / path for path in args.imported_files], project_root=root, config=config)
        print(json.dumps(result.as_dict() if hasattr(result, "as_dict") else result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "validate":
        if bool(args.validation) != bool(args.test):
            parser.error("split validation requires both --validation and --test along with train path")
        if args.validation and args.test:
            split_paths={"train":root/args.path,"validation":root/args.validation,"test":root/args.test}
            print(json.dumps(validate_splits(split_paths,temp_directory=root/"data"/"processed"),ensure_ascii=False,indent=2))
        else:
            print(json.dumps(validate_jsonl(root / args.path, require_provenance=args.require_provenance), ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(json.dumps(json.loads((root/args.path).read_text(encoding="utf-8")),ensure_ascii=False,indent=2))
    elif args.command == "readiness":
        config=load_phase8_config(); report_path=root/config["report_json"]
        report_data=json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps(report_data.get("readiness",{}),ensure_ascii=False,indent=2))
    elif args.command == "status":
        config=load_phase8_config(); store=CorpusMetadataStore(root,config["metadata_database"])
        try:
            result={"storage":store.status(),"downloads":{},"latest_readiness":None}
            for source_id in config["approved_source_ids"]:
                archive=root/config["raw_directory"]/f"{source_id}.tar.bz2"
                result["downloads"][source_id]={"present":archive.exists(),"bytes":archive.stat().st_size if archive.exists() else None}
            rp=root/config["report_json"]
            if rp.exists(): result["latest_readiness"]=json.loads(rp.read_text(encoding="utf-8")).get("readiness")
            print(json.dumps(result,ensure_ascii=False,indent=2))
        finally:
            store.close()
    elif args.command == "dashboard":
        report = json.loads((root / args.report_json).read_text(encoding="utf-8"))
        print(write_dashboard(report, StorageManager(root).resolve(args.output_html)))
    elif args.command == "benchmark":
        print(json.dumps(benchmark_jsonl(root / args.path), ensure_ascii=False, indent=2))
    elif args.command == "benchmark-source":
        print(json.dumps(benchmark_source_records(root / args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
