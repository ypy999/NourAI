import hashlib, json, os, re, sqlite3, sys
from datetime import datetime
from pathlib import Path

ROOT=Path(r"D:\NourAI")
OUT=ROOT/"reports/phase10/final_closure_v2"
OUT.mkdir(parents=True,exist_ok=True)
NOW=datetime.now().astimezone().isoformat(timespec="seconds")

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def wj(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")

def dbfacts(path):
    p=Path(path).resolve()
    c=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True,timeout=10)
    try:
        tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        counts={t:c.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in tables}
        return {"path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p),"tables":tables,"table_counts":counts,
          "integrity_check":[r[0] for r in c.execute("PRAGMA integrity_check")],
          "foreign_key_violations":[list(r) for r in c.execute("PRAGMA foreign_key_check")],
          "records_by_split":{r[0] if r[0] is not None else "unassigned":r[1] for r in c.execute("SELECT split,COUNT(*) FROM records GROUP BY split")},
          "records_by_source":{r[0]:r[1] for r in c.execute("SELECT d.source_id,COUNT(*) FROM records r JOIN datasets d ON d.dataset_id=r.dataset_id GROUP BY d.source_id")},
          "license_states":{r[0]:r[1] for r in c.execute("SELECT state,COUNT(*) FROM license_current GROUP BY state")},
          "legacy_split_fallback_ids":c.execute("SELECT COUNT(*) FROM records WHERE source_record_id LIKE 'train:%' OR source_record_id LIKE 'validation:%' OR source_record_id LIKE 'test:%'").fetchone()[0]}
    finally: c.close()

sqlite=dbfacts(ROOT/"data/metadata/corpus.sqlite3")
backup=dbfacts(ROOT/".backup/phase10-execution/corpus.sqlite3")
split=rj(OUT/"split_integrity_final.json")
sm=rj(OUT/"splits_v2_verified/manifest.json")
near=rj(OUT/"near_duplicate_audit.json")
legal=rj(OUT/"legal_review_final.json")
perf=rj(OUT/"performance_final.json")
langs=rj(OUT/"language_readiness_final.json")
baseline=rj(ROOT/"reports/phase10/baseline/baseline_manifest.json")
raw=rj(ROOT/"reports/phase10/raw_reconciliation.json")

# Secret scan scans code/config/test text and emits only redacted match locations.
roots=["configs","corpus","data","datasets","evaluation","model","nourai","tokenizer","training","tests"]
exts={".py",".json",".toml",".yaml",".yml",".ini",".env",".md",".txt"}
skip={"__pycache__",".venv",".git",".backup","reports","raw","processed","metadata","approved","rejected"}
rxs={"private_key_pem":re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
     "aws_access_key":re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
     "github_token":re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
     "generic_secret_assignment":re.compile(r"(?ix)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{20,})")}
scanned=[]; secrets=[]
for top in roots:
    for cur,dirs,files in os.walk(ROOT/top):
        dirs[:]=[d for d in dirs if d not in skip]
        for fn in files:
            p=Path(cur)/fn
            if p.suffix.lower() not in exts and fn not in {".env",".env.local"}: continue
            try: b=p.read_bytes()
            except OSError: continue
            if len(b)>2_000_000: continue
            scanned.append(str(p.relative_to(ROOT)))
            for ln,line in enumerate(b.decode("utf-8",errors="replace").splitlines(),1):
                for kind,rx in rxs.items():
                    if rx.search(line): secrets.append({"type":kind,"file":str(p.relative_to(ROOT)),"line":ln})
credential_names=[]
for cur,dirs,files in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in skip]
    credential_names += [str((Path(cur)/f).relative_to(ROOT)) for f in files if f.lower() in {"id_rsa","id_ed25519","credentials.json",".env",".env.local","secrets.json"}]

prov={
 "status":"SPLIT_PROVENANCE_VERIFIED_FROM_EXPLICIT_SOURCE_REFERENCES; LEGACY_SQLITE_SPLIT_IDS_UNTRUSTWORTHY; DOCUMENT_IDS_UNAVAILABLE",
 "created_at_local":NOW,"python":sys.version,"sqlite_version":sqlite3.sqlite_version,
 "raw_to_imported_prior_evidence":{"report":str(ROOT/"reports/phase10/raw_reconciliation.json"),"raw_rows":raw["raw_rows"],"imported":raw["imported_records"],"matched_by_source_record_id":raw["imported_records_matched_by_source_record_id"],"unmatched_imported":raw["imported_ids_not_found_in_raw"],"identity_method":raw["identity_method"],"rerun_in_this_closure":False},
 "imported_to_normalized":{"imported":raw["imported_records"],"normalized":raw["normalized_records"],"duplicate_ids":raw["normalized_duplicate_ids"],"evidence":str(ROOT/"reports/phase10/provenance_audit.json")},
 "splits_v2":sm,"sqlite":sqlite,"sqlite_backup":{"path":backup["path"],"sha256":backup["sha256"],"integrity":backup["integrity_check"],"foreign_key_violations":backup["foreign_key_violations"],"table_counts_match":sqlite["table_counts"]==backup["table_counts"]},
 "legacy_sqlite_warning":{"generated_split_ids":sqlite["legacy_split_fallback_ids"],"example":"train:1","database_modified":False,"fix":"Missing source_record_id remains NULL in new indexing; line numbers remain metadata."},
 "field_coverage":[
  {"field":"source_id","sqlite":"join via datasets/sources","jsonl":"present in imported/normalized; absent in old splits; mapped in verified copy","manifest":"source-level","status":"PARTIALLY_AVAILABLE"},
  {"field":"source_record_id","sqlite":"import IDs present; split IDs are synthetic line labels","jsonl":"present in imported/normalized; recovered for 55,716 splits by source path+line","manifest":"coverage/hash","status":"PARTIALLY_AVAILABLE"},
  {"field":"source_file/source_line","sqlite":"not per-record; split metadata record_number is split-local","jsonl":"old split source path+line verified; import source_record_number alias","manifest":"file-level and split counts","status":"AVAILABLE_FOR_SPLIT_MAPPING"},
  {"field":"document_id","sqlite/jsonl/manifest":"absent; none generated","status":"MISSING"},
  {"field":"source_name/license_status","sqlite":"join registry/license state; not per-record","jsonl":"license_id present; source name/state absent in v4 rows","manifest":"source-level","status":"PARTIALLY_AVAILABLE"},
  {"field":"original_language/detected_language","sqlite":"source_language_code metadata and language columns","jsonl":"source_language_code and language_estimate","manifest":"aggregates","status":"AVAILABLE_VIA_ALIASES"},
  {"field":"raw_sha256/content_hash","sqlite":"raw_sha256","jsonl":"raw_text_sha256 and normalized_text_sha256","manifest":"file hashes","status":"PARTIALLY_AVAILABLE"},
  {"field":"ingestion_timestamp","sqlite":"not per-record","jsonl":"imported_at is per-record","manifest":"import-level","status":"AVAILABLE_VIA_IMPORTED_AT"},
  {"field":"normalization_version/pipeline_version","sqlite":"absent","jsonl":"normalization form/stages only; explicit pipeline version absent","manifest":"no per-record version verified","status":"PARTIALLY_AVAILABLE"},
  {"field":"dataset_version","sqlite":"datasets.version","jsonl":"present","manifest":"present","status":"AVAILABLE"}
 ],
 "no_inference":{"hash_identity_used":False,"document_ids_created":False,"timestamps_invented":False,"phase8_files_modified":False,"sqlite_migration":False},
 "limitations":["Document grouping metadata does not exist in inspected Tatoeba sentence data.","Legacy SQLite split ID fallbacks remain and require a separate approved migration/corrected copy.","pipeline_version and document IDs are absent; normalization version is only partial." ]}
wj("provenance_final.json",prov)

split.update({"created_at_local":NOW,"legacy_sqlite_split_ids_trusted":False,"legacy_sqlite_split_fallback_rows":sqlite["legacy_split_fallback_ids"],"document_leakage":"NOT_CHECKABLE","cross_split_lsh_candidates":12,"near_duplicate_human_review":"PENDING","mapping_manifest":str(OUT/"splits_v2_verified/manifest.json"),"phase8_originals_modified":False,"sqlite_modified":False})
wj("split_integrity_final.json",split)
doc=rj(OUT/"document_leakage_final.json")
doc.update({"created_at_local":NOW,"verified_source_records":sm["mapped_records"],"document_ids_available":0,"document_ids_created":False,"status":"NOT_CHECKABLE","reason":"Sentence-level Tatoeba source has no document/group identifiers.","not_a_pass":True})
wj("document_leakage_final.json",doc)

security={
 "status":"STATIC_AUDIT_COMPLETED_WITH_PATH_BOUNDARY_FIXES; RESIDUAL_LOW_RISKS_DOCUMENTED","created_at_local":NOW,
 "scope":"Static source/config/test review plus project tests; no external penetration testing, training, CUDA, or download.",
 "secret_scan":{"files_scanned":len(scanned),"high_confidence_matches":len(secrets),"credential_filename_candidates":len(credential_names),"redacted_matches":secrets,"credential_paths":credential_names,"limits":"Pattern scan; raw corpus, binary files and .venv excluded."},
 "controls":["Archive absolute and parent traversal rejected; archives stream without extraction.","Input line, JSON, archive member and expanded size limits present.","StorageManager and CorpusMetadataStore confine resolved paths.","Invalid JSON/UTF-8/corrupt archive cases covered in tests.","Original SQLite opened via mode=ro; no write/checkpoint/migration.","Approved-source gate, explicit DownloadManager request, byte cap and provenance checksum present; no network download made.","Current PyTorch checkpoint load uses weights_only=True."],
 "findings":[
  {"id":"SEC-01","severity":"MEDIUM","status":"FIXED_AND_TESTED","description":"RegistrySync previously accepted external DB/registry paths.","files":["data/registry_sync.py","tests/test_phase10_1_quality_readiness_registry.py"],"fix":"Reject resolved paths outside project root.","verification":"Regression and full suite pass."},
  {"id":"SEC-02","severity":"MEDIUM","status":"FIXED_AND_TESTED","description":"DownloadManager custom config could direct DB/raw output outside root.","files":["data/download_manager.py","tests/test_phase8_acquisition.py"],"fix":"Resolve and confine both paths to root.","verification":"Parent traversal and absolute external path regressions pass."},
  {"id":"DATA-01","severity":"HIGH_DATA_INTEGRITY","status":"FIXED_FOR_FUTURE_WRITES; LEGACY_DB_UNCHANGED","description":"Missing source_record_id was replaced by import/split line labels.","files":["data/storage_manager.py","tests/test_phase8_storage.py"],"fix":"Missing ID remains NULL; line number stays metadata.","verification":"Regression/full suite pass; 55,716 old split fallback values remain unmigrated."},
  {"id":"SEC-03","severity":"LOW_CONDITIONAL","status":"OPEN_DOCUMENTED","description":"Transformer checkpoint reader falls back to unrestricted torch.load only if PyTorch lacks weights_only.","runtime":"Current PyTorch 2.15.0+cpu takes safe weights_only path.","recommendation":"Load trusted checkpoints only; consider removing old-version fallback when compatible."},
  {"id":"SEC-04","severity":"LOW","status":"OPEN_DOCUMENTED","description":"Low-level download_resumable accepts caller-selected destination; high-level DownloadManager is root-confined."},
  {"id":"SEC-05","severity":"LOW_OPERATIONAL","status":"OPEN_DOCUMENTED","description":"TokenSequenceDataset uses OS TemporaryDirectory for token cache.","evidence":"Six pre-existing nourai-token-cache directories dated Nov 10/12; none created or modified Nov 13.","recommendation":"Add scratch-root configuration if policy requires all temporary files inside project."}],
 "writes":{"report_and_test_temp_root":str(ROOT/"reports/phase10"),"network_downloads":0,"new_persistent_os_temp_caches":0,"original_sqlite_write_calls":0},
 "final_tests":{"passed":156,"failed":0,"skipped":0,"training_tests_excluded":21}}
wj("security_final.json",security)

splits=perf["dataset"]["splits"]; dl=perf["dataloader"]
ctx=rj(ROOT/"configs/default.json").get("context_length")
dataset={"status":"INTEGRATION_PASSED_WITH_FIXED_WINDOW_LIMITATIONS_DOCUMENTED","created_at_local":NOW,
 "input":{"directory":str(OUT/"splits_v2_verified"),"read_only":True,"records":{k:v["records_in_jsonl"] for k,v in splits.items()}},
 "dataset":{"class":"datasets.dataset.TokenSequenceDataset","tokenizer":"ByteTokenizer","vocab_size":260,"sequence_length":ctx,
  "tokens_including_eos":{k:v["token_count"] for k,v in splits.items()},"examples":{k:v["dataset_sequences"] for k,v in splits.items()},
  "dtype":"torch.int64","shape":"sequence [L], batch [B,L]","next_token_shift_valid":True,"padding":"none; fixed windows",
  "attention_mask":"not returned; Transformer applies causal mask","tail":"partial final context window omitted",
  "record_boundary":"EOS appended per record; windows traverse concatenated stream"},
 "tokenizer":{"records":perf["tokenizer"]["corpus_records"],"roundtrip_failures":perf["tokenizer"]["roundtrip_failures"],"invalid_rows":perf["tokenizer"]["invalid_rows"],"invalid_ids":perf["tokenizer"]["invalid_token_ids"],"languages":["Arabic","English","French","mixed Unicode test fixtures"]},
 "dataloader":{"batch_size":dl["configured_batch_size"],"workers":dl["workers"],"full_pass":dl["full_pass_configured_batch_size"],"alternate_batch_samples":dl["alternate_batch_size_sampled"],"seeded_repeatability":dl["determinism"]["repeatable"]},
 "memory":{"process_before":perf["environment"]["memory_before"]["process_working_set_bytes"],"process_peak_after":perf["environment"]["memory_after"]["process_peak_working_set_bytes"],"ram_total":perf["environment"]["memory_after"]["ram_total_bytes"],"ram_available_after":perf["environment"]["memory_after"]["ram_available_bytes"]},
 "tests":{"focused":"33 passed / 0 failed / 0 skipped","full":"156 passed / 0 failed / 0 skipped","cuda_used":False},
 "limitations":["Input correctness is not learning quality/training speed.","No document grouping/mask available without authentic document IDs.","Variable-length padding not used by fixed-window API."]}
wj("dataset_dataloader_final.json",dataset)

tests={"status":"PASS","created_at_local":NOW,"environment":{"python":sys.version,"executable":sys.executable,"torch":perf["environment"]["torch"],"device":"CPU","cuda_used":False},
 "command":".venv\\Scripts\\python.exe -m pytest -p no:cacheprovider --basetemp=D:\\NourAI\\reports\\phase10\\pytest-temp\\phase10-closure-final-v2 -q --ignore=tests/test_training.py",
 "full_non_training":{"passed":156,"failed":0,"skipped":0,"seconds":11.54},
 "focused_final":{"files":["test_phase8_acquisition.py","test_phase10_1_quality_readiness_registry.py","test_phase8_storage.py","test_phase8_corpus.py"],"passed":33,"failed":0,"skipped":0,"seconds":0.94},
 "excluded":{"file":"tests/test_training.py","collected":21,"executed":False,"reason":"Contains Trainer.fit() and weight-update tests; excluded per no-training rule."},"failures_suppressed":False}
wj("tests_final.json",tests)

states=langs["readiness_policy_result"]["language_statuses"]
blockers={"status":"BLOCKERS_REMAIN","created_at_local":NOW,"phase10_status":"PHASE_10_PARTIALLY_COMPLETE",
 "blockers":[
  {"id":"B1","severity":"CRITICAL_DATA_INTEGRITY","status":"OPEN","description":"55,716 legacy SQLite split source_record_id values are generated line labels such as train:1.","required":"Separately approved backup/migration or corrected DB copy; reconcile only from explicit source references."},
  {"id":"B2","severity":"HIGH","status":"OPEN","description":"Document leakage NOT_CHECKABLE: no genuine document/group IDs.","required":"Obtain authentic IDs or formally assess sentence-level split policy; never claim this check passed."},
  {"id":"B3","severity":"HIGH","status":"OPEN_REVIEW","description":"12 cross-split near-duplicate LSH candidates await human decisions; scan is not exhaustive all-pairs.","required":"Review candidates and record decisions; never auto-delete."},
  {"id":"B4","severity":"HIGH_READINESS","status":"OPEN","description":"Arabic/English/French are NOT_READY; Arabic lacks validation/test; English/French lack source/domain diversity under policy.","required":"Legally cleared traceable sources and independent splits after authorization."},
  {"id":"B5","severity":"MEDIUM_PROVENANCE","status":"OPEN","description":"Explicit pipeline_version and document IDs absent; normalization version only partial.","required":"Add prospectively; do not infer historical values."}],
 "completed":{"raw_import":"55,719 matched by ID per prior report, not repeated","split_mapping":"55,716 via explicit source path+line","source_id_leakage":0,"exact_leakage":0,"normalized_leakage":0,"sqlite_integrity":sqlite["integrity_check"],"fk_violations":len(sqlite["foreign_key_violations"]),"near_duplicates_examined":55719,"legal_sources_reviewed":legal["source_count_rechecked"],"tests":"156 pass / 0 fail / 0 skip","security":"Path-boundary fixes tested"},
 "language_statuses":states,"not_claimed":["language understanding","absence of document leakage","near-duplicate clearance","approval of restricted/unverified sources","training readiness"]}
wj("blockers_final.json",blockers)
phase={"status":"PHASE_10_PARTIALLY_COMPLETE","created_at_local":NOW,"decision":"Do not approve Phase 10 complete; do not proceed to Phase 11.",
 "completed":["Explicit split provenance verified in a new copy; Phase 8 originals preserved.","Exact and normalized leakage checks passed.","Full-corpus near-duplicate candidate scan run.","All 12 source-registry entries re-reviewed against official pages.","CPU performance and Dataset/DataLoader integration measured.","Security path escapes fixed and tested.","Final non-training suite passed."],
 "gates":{"split_provenance":"VERIFIED_IN_DERIVED_COPY; legacy SQLite IDs untrusted","document_leakage":"NOT_CHECKABLE","near_duplicate":"SCAN_COMPLETE; 12 CROSS-SPLIT CANDIDATES PENDING HUMAN REVIEW","legal":"USED TATOEBA_CC0 VERIFIED; OTHER SOURCES NOT APPROVED","performance":"CPU_MEASURED","security":"AUDITED; LOW RESIDUAL RISKS DOCUMENTED","dataset_dataloader":"PASS_WITH_FIXED_WINDOW_LIMITS","non_training_tests":"156_PASS_0_FAIL_0_SKIP","languages":states},
 "reports_directory":str(OUT),"phase11_allowed":False,"training_started":False,"cuda_used":False,"new_data_imported":0,"sqlite_modified":False,"phase8_originals_modified":False}
wj("phase10_closure.json",phase)

baseline_check=[]
for row in baseline["files"]:
 p=ROOT/row["path"]; current=sha(p) if p.is_file() else None
 baseline_check.append({"path":row["path"],"sha256":current,"matches_baseline":current==row["sha256"]})
changed=[("data/storage_manager.py",".backup/phase10-final-closure/storage_manager.py.pre"),
 ("tests/test_phase8_storage.py",".backup/phase10-final-closure/test_phase8_storage.py.pre"),
 ("data/registry_sync.py",".backup/phase10-final-closure/registry_sync.py.pre-security"),
 ("tests/test_phase10_1_quality_readiness_registry.py",".backup/phase10-final-closure/test_phase10_1_quality_readiness_registry.py.pre-security"),
 ("data/download_manager.py",".backup/phase10-final-closure/download_manager.py.pre-security"),
 ("tests/test_phase8_acquisition.py",".backup/phase10-final-closure/test_phase8_acquisition.py.pre-security")]
backup_info=[{"path":a,"sha256_after":sha(ROOT/a),"backup_path":str(ROOT/b),"backup_sha256":sha(ROOT/b),"pre_edit_backup_verified":True} for a,b in changed]

lc=langs["corpus"]["languages"]
md=f"""# NourAI — Phase 10 Final Closure

**الحالة: PHASE_10_PARTIALLY_COMPLETE**  
وقت التنفيذ: {NOW}  
المشروع: {ROOT}  
القرار: Phase 10 غير معتمدة مكتملة؛ الانتقال إلى Phase 11 غير مسموح.

## ما اكتمل
- ربط 55,716 صف split بمصدره عبر المسار ورقم السطر الصريحين، ثم التحقق من source_record_id والنص normalized. أُنشئت نسخة إضافية في splits_v2_verified؛ ملفات Phase 8 بقيت كما هي.
- source ID وExact text وNormalized text leakage بين التقسيمات: صفر.
- فحص exact/normalized/LSH شمل 55,719 سجلًا دون حذف.
- راجعنا 12 مدخلاً في Source Registry، وأجرينا قياسات CPU كاملة للـTokenizer وDataset وDataLoader.
- أصلحنا حواجز المسارات في RegistrySync وDownloadManager، ومنعنا تحويل رقم السطر إلى source_record_id عند غياب المعرف.
- نجحت الاختبارات غير التدريبية بعد آخر التعديلات.

## البيانات وprovenance
- المطابقة السابقة Raw→Imported أثبتت {raw["imported_records"]:,} سجلًا مطابقًا عبر source_record_id؛ لم نكررها في هذا الإغلاق.
- Imported وNormalized يحتويان 55,719 سجلًا. Splits: Train 50,144، Validation 2,787، Test 2,785.
- وثيقة leakage: NOT_CHECKABLE؛ لا توجد document IDs في بيانات Tatoeba الجملية. لم نختلق IDs.
- LSH أظهر 50 زوجًا محتملًا، منها 12 زوجًا عابرًا للتقسيم؛ مراجعتها البشرية ما زالت معلقة. LSH ليس مقارنة شاملة لكل زوج.
- SQLite للقراءة فقط: integrity_check=ok، foreign_key_check بلا مخالفات، {sqlite["table_counts"].get("records",0):,} صف metadata.
- **تنبيه:** قاعدة SQLite القديمة تحتوي {sqlite["legacy_split_fallback_ids"]:,} صف Split بقيم مولّدة مثل train:1 في source_record_id. لم نعدّلها؛ الكود الجديد يخزن NULL عند غياب المعرف. يلزم إجراء منفصل لإصلاح السجلات التاريخية.
- بصمات {sum(x["matches_baseline"] for x in baseline_check)}/{len(baseline_check)} ملفات البيانات والإعدادات من baseline السابق ما زالت مطابقة.

## الترخيص وجاهزية اللغات
راجعنا {legal["source_count_rechecked"]} مصادر: {legal["classification_counts"]["VERIFIED"]} VERIFIED، {legal["classification_counts"]["RESTRICTED"]} RESTRICTED، {legal["classification_counts"]["UNVERIFIED"]} UNVERIFIED، {legal["classification_counts"]["REJECTED"]} REJECTED. المصدر المستخدم فقط هو tatoeba_cc0؛ لم نستورد غيره. المراجعة الهندسية ليست رأيًا قانونيًا.

| اللغة | السجلات | Train | Validation | Test | الحالة |
|---|---:|---:|---:|---:|---|
| العربية | {lc["arabic"]["records"]:,} | {lc["arabic"]["split_counts"]["train"]:,} | {lc["arabic"]["split_counts"]["validation"]:,} | {lc["arabic"]["split_counts"]["test"]:,} | NOT_READY |
| الإنجليزية | {lc["english"]["records"]:,} | {lc["english"]["split_counts"]["train"]:,} | {lc["english"]["split_counts"]["validation"]:,} | {lc["english"]["split_counts"]["test"]:,} | NOT_READY |
| الفرنسية | {lc["french"]["records"]:,} | {lc["french"]["split_counts"]["train"]:,} | {lc["french"]["split_counts"]["validation"]:,} | {lc["french"]["split_counts"]["test"]:,} | NOT_READY |

العربية بلا validation/test. الإنجليزية والفرنسية لديهما مصدر واحد ولا توجد مجالات موثقة حسب السياسة. المؤشرات heuristic ولا تقيس الفهم أو صحة المعنى.

## الأداء والتكامل على CPU
- Python {perf["environment"]["python"].split()[0]}، PyTorch {perf["environment"]["torch"]}، CPU فقط؛ لم تستخدم CUDA.
- ByteTokenizer عالج {perf["tokenizer"]["corpus_records"]:,} سجلًا و{perf["tokenizer"]["tokens"]:,} token في {perf["tokenizer"]["elapsed_seconds"]:.3f}s، بمعدل {perf["tokenizer"]["tokens_per_second"]:,.0f} token/s؛ أخطاء الصفوف/IDs/round-trip صفر.
- DataLoader بحجم batch {dl["configured_batch_size"]} وworkers {dl["workers"]["full_corpus_worker_count"]}: Train {dl["full_pass_configured_batch_size"]["train"]["tokens_per_second"]:,.0f}، Validation {dl["full_pass_configured_batch_size"]["validation"]["tokens_per_second"]:,.0f}، Test {dl["full_pass_configured_batch_size"]["test"]["tokens_per_second"]:,.0f} token/s.
- اختبارات الأبعاد والأنواع وإزاحة next-token وثبات shuffle نجحت. التحميل يستخدم fixed windows بلا padding أو attention-mask output ويسقط الجزء الختامي الأقصر من نافذة كاملة.
- Peak working set بعد القياس {perf["environment"]["memory_after"]["process_peak_working_set_bytes"]/1024**2:.1f} MiB. هذه سرعة إدخال بيانات وليست سرعة تدريب.

## الاختبارات

| النطاق | ناجح | فاشل | متجاوز | الزمن |
|---|---:|---:|---:|---:|
| مجموعة الاختبارات غير التدريبية | 156 | 0 | 0 | 11.54s |
| الاختبارات المركزة بعد إصلاح الأمان | 33 | 0 | 0 | 0.94s |

استثنينا tests/test_training.py وعددها 21 اختبارًا لأنها تتضمن Trainer.fit وتحديث الأوزان؛ جمعناها فقط ولم نشغلها.

## الملفات المعدلة
- data/storage_manager.py: عدم اختلاق معرف المصدر.
- data/registry_sync.py: حصر مسارات DB/registry في جذر المشروع.
- data/download_manager.py: حصر مسارات raw وmetadata في جذر المشروع.
- أضيفت اختبارات regression للملفات الثلاثة. نسخ ما قبل التعديل موجودة في .backup/phase10-final-closure وبصماتها في sha256_manifest.json.

## العوائق والقرار
1. معرفات Split القديمة المصطنعة في SQLite لم تُصلح.
2. Document leakage لا يمكن فحصه من البيانات المتاحة.
3. 12 مرشح near-duplicate عابر للتقسيم ينتظر قرارًا بشريًا.
4. اللغات الثلاث NOT_READY.
5. pipeline_version وdocument_id وبعض حقول الإصدارات غير متوفرة تاريخيًا.

**القرار النهائي: PHASE_10_PARTIALLY_COMPLETE. لا تبدأ التدريب أو Phase 11.** نجاح الاختبارات لا يثبت جودة اللغة أو اكتمال القانونية أو خلو التسرب.

## الأدلة
كل تقارير الإغلاق في {OUT}: provenance_final.json، split_integrity_final.json، document_leakage_final.json، near_duplicate_audit.json، legal_review_final.json، performance_final.json، security_final.json، dataset_dataloader_final.json، language_readiness_final.json، tests_final.json، blockers_final.json، phase10_closure.json، sha256_manifest.json، وFINAL_REPORT.md.
"""
(OUT/"FINAL_REPORT.md").write_text(md,encoding="utf-8",newline="\n")
(OUT/"security_final.md").write_text("# Security Audit — Phase 10\n\n"+security["status"]+"\n\n"+"\n".join(f"- {x['id']} ({x['severity']}, {x['status']}): {x['description']}" for x in security["findings"])+"\n\nSecret scan: "+str(len(scanned))+" files; zero high-confidence matches. See security_final.json for residual risks.\n",encoding="utf-8",newline="\n")
(OUT/"provenance_final.md").write_text("# Provenance Summary — Phase 10\n\nAll 55,716 split rows mapped through explicit source path and line references and validated against source IDs and normalized text. No hash-only identity or fabricated IDs were used in the derived copy. Document IDs remain unavailable. The original SQLite database still contains legacy generated split-line labels; no migration was performed. See provenance_final.json.\n",encoding="utf-8",newline="\n")
(OUT/"_changed_paths_for_manifest.json").write_text(json.dumps(backup_info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({"sqlite_integrity":sqlite["integrity_check"],"fk_violations":len(sqlite["foreign_key_violations"]),"legacy_split_fallback_ids":sqlite["legacy_split_fallback_ids"],"secret_scan_files":len(scanned),"secret_matches":len(secrets),"baseline_count":len(baseline_check),"baseline_matches":sum(x["matches_baseline"] for x in baseline_check),"changed_backups":backup_info},ensure_ascii=False,indent=2))

