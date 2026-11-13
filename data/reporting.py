"""JSON and Markdown reports for dataset engineering runs."""
from datetime import datetime
import ctypes
import json
import math
import os
from pathlib import Path
import tempfile
import sys


def memory_snapshot():
    """Best-effort Windows memory snapshot; missing APIs return null fields."""
    result = {"total_ram_bytes": None, "available_ram_bytes": None,
              "process_working_set_bytes": None, "process_peak_working_set_bytes": None}
    if os.name != "nt":
        return result
    try:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["total_ram_bytes"] = int(status.ullTotalPhys)
            result["available_ram_bytes"] = int(status.ullAvailPhys)
    except Exception:
        pass
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            result["process_working_set_bytes"] = int(counters.WorkingSetSize)
            result["process_peak_working_set_bytes"] = int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return result


def _plain(value):
    if hasattr(value, "as_dict"):
        return _plain(value.as_dict())
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_dataset_report(*, dataset_overview=None, source=None, license=None,
                         language_statistics=None, cleaning_statistics=None,
                         duplicate_statistics=None, quality_statistics=None,
                         tokenizer_statistics=None, split_statistics=None,
                         storage_information=None, processing_seconds=None,
                         memory_measurements=None, warnings=None, errors=None,
                         readiness=None, test_results=None):
    report = {
        "schema_version": 1,
        "report_type": "dataset_engineering",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": {"python": sys.version, "platform": sys.platform},
        "dataset_overview": dataset_overview or {"record_count": 0, "corpus_present": False},
        "source": source,
        "license": license,
        "language_distribution": language_statistics,
        "cleaning_statistics": cleaning_statistics,
        "duplicate_statistics": duplicate_statistics,
        "quality_statistics": quality_statistics,
        "tokenizer_statistics": tokenizer_statistics,
        "split_statistics": split_statistics,
        "storage_information": storage_information,
        "processing_seconds": processing_seconds,
        "memory_measurements": memory_measurements if memory_measurements is not None else memory_snapshot(),
        "warnings": warnings or [],
        "errors": errors or [],
        "readiness_assessment": readiness or {"eligible": False, "blockers": ["no user dataset was provided"]},
        "test_results": test_results,
        "limitations": ["Heuristic language labels are approximate.",
                        "Quality scores measure format/integrity signals, not factual accuracy or educational value.",
                        "Tokenizer analysis measures byte counts, not linguistic segmentation quality."],
    }
    return _plain(report)


def _markdown(report):
    overview = report.get("dataset_overview") or {}
    ready = report.get("readiness_assessment") or {}
    lines = ["# NourAI Dataset Engineering Report", "",
             f"- Created: {report.get('created_at')}",
             f"- Dataset: {overview.get('name', 'not provided')}",
             f"- Records: {overview.get('record_count', 0)}",
             f"- Corpus present: {overview.get('corpus_present', False)}",
             f"- Training eligible: {ready.get('eligible', False)}", "",
             "## Provenance", "",
             f"- Source: {json.dumps(report.get('source'), ensure_ascii=False)}",
             f"- License: {json.dumps(report.get('license'), ensure_ascii=False)}", "",
             "## Results", ""]
    for label, key in (("Language distribution", "language_distribution"),
                       ("Cleaning statistics", "cleaning_statistics"),
                       ("Duplicate statistics", "duplicate_statistics"),
                       ("Quality statistics", "quality_statistics"),
                       ("Tokenizer statistics", "tokenizer_statistics"),
                       ("Train/validation/test splits", "split_statistics"),
                       ("Storage", "storage_information"),
                       ("Memory measurements", "memory_measurements"),
                       ("Test results", "test_results")):
        lines.extend([f"### {label}", "", "```json", json.dumps(report.get(key), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Processing time", "", str(report.get("processing_seconds")), "",
                  "## Warnings and errors", ""])
    lines.extend([f"- Warning: {item}" for item in report.get("warnings", [])] or ["- No warnings recorded"])
    lines.extend([f"- Error: {item}" for item in report.get("errors", [])] or ["- No errors recorded"])
    lines.extend(["", "## Readiness blockers", ""])
    lines.extend([f"- {item}" for item in ready.get("blockers", [])] or ["- None"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def write_dataset_reports(report, output_dir, stem="dataset_report"):
    """Write JSON and Markdown report files without overwriting existing files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": output_dir / f"{stem}.json", "markdown": output_dir / f"{stem}.md"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("dataset report exists; choose a new report stem")
    temp = {}
    try:
        for key, destination in paths.items():
            fd, name = tempfile.mkstemp(prefix=f"{stem}-", suffix=".tmp", dir=output_dir)
            os.close(fd)
            temp[key] = Path(name)
            with temp[key].open("w", encoding="utf-8", newline="\n") as output:
                if key == "json":
                    json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
                    output.write("\n")
                else:
                    output.write(_markdown(report))
                output.flush()
                os.fsync(output.fileno())
        if any(path.exists() for path in paths.values()):
            raise FileExistsError("dataset report was created concurrently; refusing overwrite")
        temp["json"].replace(paths["json"])
        del temp["json"]
        temp["markdown"].replace(paths["markdown"])
        del temp["markdown"]
    finally:
        for path in temp.values():
            if path.exists():
                path.unlink()
    return {key: str(path) for key, path in paths.items()}
