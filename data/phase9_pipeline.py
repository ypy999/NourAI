"""Streaming Phase 9 analysis over the Phase 8 corpus without rewriting it."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ctypes
import json
from pathlib import Path
import time

from data.domain_analysis import analyze_arabic_text, classify_domains
from data.language_balancing import analyze_language_balance, validate_target_ratios
from data.phase9_readiness import assess_phase9_readiness
from data.language_analysis import classify_language
from data.quality import score_record
from corpus.validation import validate_splits
from tokenizer.byte_tokenizer import ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_CODES = {"ara": "arabic", "eng": "english", "fra": "french"}


def _iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _process_memory_bytes():
    """Return current/peak working set on Windows, or None if unavailable."""
    try:
        class _Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        memory_api = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_api.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        memory_api.restype = ctypes.c_int
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if memory_api(process, ctypes.byref(counters), counters.cb):
            return {"working_set_bytes": counters.WorkingSetSize,
                    "peak_working_set_bytes": counters.PeakWorkingSetSize}
    except (AttributeError, OSError):
        pass
    return None


def analyze_corpus(imported_path, split_paths, *, registry, tokenizer=None):
    """Compute corpus and split diagnostics in a bounded-memory pass."""
    tokenizer = tokenizer or ByteTokenizer()
    lang_stats = defaultdict(lambda: {"records": 0, "tokens_including_eos": 0, "utf8_bytes": 0,
                                      "characters": 0, "sources": set(), "domains": Counter(),
                                      "quality_distribution": Counter(), "quality_sum": 0.0,
                                      "quality_scored": 0, "empty_records": 0, "arabic_diagnostics": Counter()})
    duplicate_counts = Counter()
    text_language_counts = Counter()
    for row in _iter_jsonl(imported_path):
        text = row.get("cleaned_text", row.get("raw_text", row.get("text", "")))
        if not isinstance(text, str):
            continue
        code = row.get("source_language_code", "und")
        language = LANGUAGE_CODES.get(code, "unknown")
        text_language_counts[classify_language(text)] += 1
        domains = classify_domains(text, source_domains=row.get("sentence_domain"))
        score = score_record({"text": text})
        tokens = tokenizer.encode(text, add_eos=True)
        item = lang_stats[language]
        item["records"] += 1
        item["tokens_including_eos"] += len(tokens)
        item["utf8_bytes"] += len(text.encode("utf-8"))
        item["characters"] += len(text)
        item["sources"].add(row.get("source_id", "unknown"))
        item["domains"].update(domains["domains"])
        item["quality_sum"] += float(score.get("quality_score", score.get("score", 0)))
        item["quality_scored"] += 1
        item["empty_records"] += not bool(text.strip())
        if language == "arabic":
            item["arabic_diagnostics"].update(analyze_arabic_text(text))
        duplicate_counts["exact"] += bool(row.get("is_exact_duplicate"))
        duplicate_counts["normalized"] += bool(row.get("is_normalized_duplicate"))
    # A second streaming pass feeds the reusable language-balance component.
    phase9_config = json.loads((PROJECT_ROOT / "configs/phase9_corpus.json").read_text(encoding="utf-8-sig"))
    balance = analyze_language_balance(_balance_records(imported_path, tokenizer),
                                       target_ratios=validate_target_ratios(phase9_config["target_language_ratios"]),
                                       tokenizer=tokenizer)
    split_stats = defaultdict(lambda: defaultdict(lambda: {"records": 0, "tokens": 0, "bytes": 0}))
    for split, path in split_paths.items():
        for row in _iter_jsonl(path):
            text = row.get("cleaned_text", row.get("raw_text", row.get("text", "")))
            language = classify_language(text)
            stats = split_stats[language][split]
            stats["records"] += 1
            stats["tokens"] += len(tokenizer.encode(text, add_eos=True))
            stats["bytes"] += len(text.encode("utf-8"))
    languages = {}
    for name, stats in lang_stats.items():
        stats["mean_characters"] = stats["characters"] / stats["records"] if stats["records"] else None
        stats["unique_sources"] = len(stats["sources"])
        stats["sources"] = sorted(stats["sources"])
        stats["unique_heuristic_domains"] = len(stats["domains"])
        stats["domain_counts"] = dict(stats["domains"])
        stats["heuristic_domains"] = sorted(stats["domains"])
        stats["mean_quality_score"] = stats["quality_sum"] / stats["quality_scored"] if stats["quality_scored"] else None
        stats["quality_distribution"] = dict(stats["quality_distribution"])
        stats["arabic_diagnostics"] = dict(stats["arabic_diagnostics"])
        del stats["quality_sum"], stats["quality_scored"], stats["domains"]
        del stats["sources"]
        languages[name] = stats
    split_languages = {}
    for lang, splits in split_stats.items():
        split_languages[lang] = {f"{split}_{metric}": value for split, row in splits.items() for metric, value in row.items()}
    split_validation = validate_splits(split_paths)
    leak_count = split_validation.get("duplicate_leakage")
    source_catalog = registry.list_sources()
    used_ids = {"tatoeba_cc0"} if any(s["records"] for s in languages.values()) else set()
    source_map = {source["source_id"]: source for source in source_catalog}
    used_sources = [source_map[sid] for sid in used_ids if sid in source_map]
    readiness = assess_phase9_readiness(language_statistics=balance["languages"],
                                        split_language_statistics=split_languages, used_sources=used_sources,
                                        split_validation=split_validation, balance_statistics=balance,
                                        min_eval_records=phase9_config["minimum_evaluation_records_per_split"],
                                        min_eval_tokens=phase9_config["minimum_evaluation_tokens_per_split"],
                                        min_sources_per_language=phase9_config["minimum_sources_per_language"],
                                        min_verified_domains_per_language=phase9_config["minimum_source_verified_domains_per_language"],
                                        training_allowed=phase9_config["training_allowed_in_phase9"])
    total_records = sum(v["records"] for v in languages.values())
    text_language_rates = {key: {"records": value, "fraction": value / total_records if total_records else 0.0}
                           for key, value in sorted(text_language_counts.items())}
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "source_corpus": str(imported_path),
            "records": sum(v["records"] for v in languages.values()), "languages": languages,
            "text_language_classification": text_language_rates,
            "splits": split_languages, "split_hash_leakage": leak_count,
            "split_validation": split_validation,
            "duplicate_counts_import_metadata": dict(duplicate_counts), "balance": balance,
            "readiness": readiness, "source_count": len(used_sources), "sources": used_sources,
            "training_started": False, "new_corpus_imported": False,
            "limitations": ["Tatoeba corpus has no domain metadata; domain labels are heuristic.",
                            "Language counts rely on source language codes and are not independently verified.",
                            "This diagnostic does not demonstrate linguistic competence."]}


def _balance_records(path, tokenizer):
    for row in _iter_jsonl(path):
        text = row.get("cleaned_text", row.get("raw_text", row.get("text", "")))
        code = row.get("source_language_code", "und")
        domain = classify_domains(text)
        heuristic_language = classify_language(text)
        if heuristic_language == "unclassified":
            heuristic_language = "unknown"
        yield {"raw_text": text, "language": heuristic_language,
               "token_count": len(tokenizer.encode(text, add_eos=True)), "source_id": row.get("source_id"),
               "domains": domain["domains"], "domain_method": domain["method"],
               "quality_score": score_record({"text": text})["score"]}


def write_phase9_reports(result, reports_dir, *, project_root=PROJECT_ROOT):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase9_corpus_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    balance = result["balance"]
    (reports_dir / "phase9_language_balance.json").write_text(json.dumps(balance, ensure_ascii=False, indent=2), encoding="utf-8")
    domain_distribution = {language: stats.get("domain_counts", {}) for language, stats in result["languages"].items()}
    (reports_dir / "phase9_domain_distribution.json").write_text(json.dumps(domain_distribution, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "phase9_benchmark.json").write_text(json.dumps(result.get("scaling_benchmark", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    gaps = {"status": result["readiness"]["status"], "blockers": result["readiness"]["all_blockers"],
            "language_records": {k: v["records"] for k, v in result["languages"].items()},
            "measured_language_gaps": result.get("measured_language_gaps", {}),
            "acquisition_needed": ["Arabic: obtain an authorized Common Voice Arabic 27 archive and ingest only validated sentence rows after terms review.",
                                   "English/French: obtain independent licensed general-text sources; identified Common Voice speech subsets are not sufficient for general-domain LLM data."],
            "no_data_acquired": not result["new_corpus_imported"]}
    (reports_dir / "phase9_gap_analysis.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "\n".join(f"| {lang} | {stats['records']:,} | {stats['tokens_including_eos']:,} | {stats['utf8_bytes']:,} | {stats['mean_characters']:.2f} |" for lang, stats in result["languages"].items())
    split_totals = {name: {metric: sum(stats.get(f"{name}_{metric}", 0) for stats in result["splits"].values())
                           for metric in ("records", "tokens", "bytes")} for name in ("train", "validation", "test")}
    split_rows = "\n".join(f"| {name} | {totals['records']:,} | {totals['tokens']:,} | {totals['bytes']:,} |" for name, totals in split_totals.items())
    domain_lines = "\n".join(f"- **{language}:** " + ", ".join(f"{domain}={count:,}" for domain, count in sorted(stats.get("domain_counts", {}).items())) for language, stats in result["languages"].items())
    license_lines = "\n".join(f"- [{item['source_id']}]({item.get('url', '')}): {item.get('license_id')} / {item.get('status')}, approved={item.get('approved_for_ingestion')}. {item.get('legal_review_note', item.get('notes', ''))}" for item in result.get("source_candidates", []))
    prior_langs = result.get("phase8_reference", {}).get("quality", {}).get("language_counts_heuristic", {})
    prior_total = sum(prior_langs.values())
    mixed_unknown = {name: {"records": count, "fraction": count / prior_total if prior_total else 0.0}
                     for name, count in prior_langs.items() if name in ("mixed", "unclassified")}
    md = ("# Phase 9 Corpus and Gap Analysis\n\n"
          f"Generated: {result['generated_at']}\n\nStatus: **{result['readiness']['status']}**. No training started; no new corpus imported.\n\n"
          "## Current corpus\n\n| Language | Records | ByteTokenizer IDs incl. EOS | UTF-8 text bytes | Mean characters |\n|---|---:|---:|---:|---:|\n" + rows +
          "\n\n## Readiness blockers\n\n" + "\n".join(f"- {item}" for item in result['readiness']['all_blockers']) +
          "\n\n## Heuristic domain distribution\n\n" + domain_lines +
          "\n\n## Phase 8 text-level mixed and unclassified rates\n\n" +
          "\n".join(f"- {name}: {row['records']:,} ({row['fraction']:.2%})" for name, row in mixed_unknown.items()) +
          "\n\n## Existing splits (text-derived language labels)\n\n| Split | Records | ByteTokenizer IDs incl. EOS | UTF-8 bytes |\n|---|---:|---:|---:|\n" + split_rows +
          f"\n\nNormalized duplicates removed before splitting: {result.get('normalized_duplicates_removed_before_split', 'unknown')}; exact duplicates: {result.get('exact_duplicates_before_split', 'unknown')}; normalized duplicate fraction: {result.get('duplicate_fraction_before_split', 'unknown')}; duplicate leakage: {result['split_hash_leakage']}; validation: {result['split_validation']['valid']}.\n" +
          f"\nPhase 7 format/integrity quality mean: {result.get('phase8_reference', {}).get('quality', {}).get('mean_record_score', 'unknown')}; interpretation: heuristic, not semantic or factual quality.\n" +
          "\n## Source candidates and license gates\n\n" + license_lines +
          "\n\nHeuristic domain assignment is not validated labeling; corpus size and passing software checks do not establish language capability.\n")
    (reports_dir / "phase9_corpus_report.md").write_text(md, encoding="utf-8")
    (reports_dir / "phase9_gap_analysis.md").write_text(md, encoding="utf-8")
    (reports_dir / "phase9_corpus_dashboard.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>NourAI Phase 9</title><h1>Phase 9 corpus dashboard</h1>"
        f"<p>Status: {result['readiness']['status']}</p><h2>Languages</h2><pre>{json.dumps(result['languages'], ensure_ascii=False, indent=2)}</pre>"
        f"<h2>Splits</h2><pre>{json.dumps(result['splits'], ensure_ascii=False, indent=2)}</pre>"
        f"<h2>Source license gates</h2><pre>{json.dumps(result.get('source_candidates', []), ensure_ascii=False, indent=2)}</pre>"
        f"<h2>Readiness</h2><pre>{json.dumps(result['readiness'], ensure_ascii=False, indent=2)}</pre>", encoding="utf-8")
    report = "# PHASE 9 — Multilingual Corpus Expansion & Balancing\n\n" + md + "\n## Data governance\n\nNo candidate source was downloaded or ingested in this run. Candidate access/license restrictions are documented in the source registry. Heuristic domains are discovery-only. Training was not started.\n"
    (Path(project_root) / "PHASE_9_MULTILINGUAL_CORPUS_REPORT.md").write_text(report, encoding="utf-8")
    return [reports_dir / name for name in ("phase9_corpus_report.json", "phase9_language_balance.json", "phase9_domain_distribution.json", "phase9_benchmark.json", "phase9_gap_analysis.json", "phase9_corpus_report.md", "phase9_gap_analysis.md", "phase9_corpus_dashboard.html")]


def run_phase9(project_root=PROJECT_ROOT, *, include_benchmark=True):
    project_root = Path(project_root)
    imported = project_root / "data/processed/phase8/imported/tatoeba_cc0_multilingual_v4.jsonl"
    splits_dir = project_root / "data/processed/phase8/deduplicated/splits"
    splits = {name: splits_dir / f"{name}.jsonl" for name in ("train", "validation", "test")}
    from data.source_registry import SourceRegistry
    registry = SourceRegistry(project_root / "configs/dataset_sources.json")
    started = time.perf_counter()
    memory_before = _process_memory_bytes()
    result = analyze_corpus(imported, splits, registry=registry)
    result["analysis_performance"] = {"elapsed_seconds": time.perf_counter() - started,
                                      "process_memory_before_bytes": memory_before,
                                      "process_memory_after_bytes": _process_memory_bytes(),
                                      "throughput_records_per_second": result["records"] / max(time.perf_counter() - started, 1e-12),
                                      "hardware": "CPU; process working-set measurements via Windows API when available"}
    benchmark_path = project_root / "reports/phase9_benchmark.json"
    if include_benchmark:
        from data.phase9_benchmark import run_scaling_benchmark
        result["scaling_benchmark"] = run_scaling_benchmark()
    elif benchmark_path.is_file():
        result["scaling_benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        result["scaling_benchmark"] = {"one_million_run": "not run; no saved benchmark exists", "results": []}
    result["source_candidates"] = [source for source in registry.list_sources() if source["source_id"] != "tatoeba_cc0"]
    prior_path = project_root / "reports/phase8_corpus_readiness.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    result["phase8_reference"] = {"quality": prior["quality"], "deduplication": prior["validation"]["pre_dedup_import"],
                                  "deduplicated_splits": prior["validation"]["deduplicated_splits"],
                                  "dashboard": prior["corpus_dashboard_metrics"]}
    result["text_language_classification_phase8"] = prior["quality"]["language_counts_heuristic"]
    result["normalized_duplicates_removed_before_split"] = prior["validation"]["deduplicated_splits"]["normalized_duplicates_removed_before_split"]
    result["exact_duplicates_before_split"] = 0
    result["duplicate_fraction_before_split"] = result["normalized_duplicates_removed_before_split"] / max(result["records"], 1)
    result["measured_language_gaps"] = {language: {"observed_records": result["languages"].get(language, {}).get("records", 0),
                                                    "minimum_per_independent_split": result["readiness"]["linguistic_readiness"]["configured_minimum_eval_records_per_split"],
                                                    "note": "Actual split-level counts and tokens are listed in language readiness."}
                                       for language in ("arabic", "english", "french")}
    write_phase9_reports(result, project_root / "reports", project_root=project_root)
    return result
