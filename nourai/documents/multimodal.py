"""Mission 4 — Modular Multimodal Document Intelligence (TXT, MD, JSON, CSV, DOCX, PDF, Images/OCR hook).

Uses pure Python standard library with optional OCR/PDF library detection and graceful fallback.
Never claims visual understanding unless an actual verified OCR/vision backend is available.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3
import struct
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET
import zipfile
import zlib

from nourai.capabilities import detect_text_language
from nourai.memory.advanced_memory import _tokenize

MAX_DOC_BYTES = 8 * 1024 * 1024
MAX_DOCX_XML_BYTES = 12 * 1024 * 1024


def detect_ocr_capabilities() -> dict[str, Any]:
    """Honestly report whether optional OCR / vision libraries are installed."""
    pil_ok = False
    tess_ok = False
    pypdf_ok = False
    try:
        import PIL  # noqa: F401
        pil_ok = True
    except ImportError:
        pass
    try:
        import pytesseract  # noqa: F401
        tess_ok = True
    except ImportError:
        pass
    try:
        import pypdf  # noqa: F401
        pypdf_ok = True
    except ImportError:
        pass
    return {
        "pil_available": pil_ok,
        "pytesseract_available": tess_ok,
        "ocr_ready": bool(pil_ok and tess_ok),
        "pypdf_available": pypdf_ok,
        "vision_model_available": False,
    }


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    doc_type: str
    sha256: str
    language: str
    text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        if "|" in lines[i] and re.match(r"^\s*\|?[\s:\-]+\|[\s:\-|]+$", lines[i + 1]):
            headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            tables.append({"source": "markdown", "headers": headers, "rows": rows})
        else:
            i += 1
    return tables


def _extract_docx_text_and_tables(raw_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        info = zf.getinfo("word/document.xml")
        if info.file_size > MAX_DOCX_XML_BYTES:
            raise ValueError("decompressed .docx content exceeds safety limit")
        xml_bytes = zf.read("word/document.xml")
        if len(xml_bytes) > MAX_DOCX_XML_BYTES:
            raise ValueError("decompressed .docx content exceeds safety limit")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for p in root.iter():
        if p.tag.endswith("}p"):
            pts = [el.text for el in p.iter() if el.tag.endswith("}t") and el.text]
            if pts:
                paragraphs.append("".join(pts).strip())
    tables: list[dict[str, Any]] = []
    for tbl in root.iter():
        if tbl.tag.endswith("}tbl"):
            t_rows: list[list[str]] = []
            for tr in tbl:
                if tr.tag.endswith("}tr"):
                    cells = []
                    for tc in tr:
                        if tc.tag.endswith("}tc"):
                            cell_txt = " ".join(el.text for el in tc.iter() if el.tag.endswith("}t") and el.text).strip()
                            cells.append(cell_txt)
                    if cells:
                        t_rows.append(cells)
            if t_rows:
                tables.append({"source": "docx", "headers": t_rows[0], "rows": t_rows[1:]})
    return "\n".join(paragraphs).strip(), tables


def _extract_pdf_text(raw_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Extract text from PDF using pypdf if available, else stdlib FlateDecode + UTF-16BE Arabic hex fallback."""
    caps = detect_ocr_capabilities()
    if caps["pypdf_available"]:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages).strip(), {"parser": "pypdf", "page_count": len(reader.pages)}

    parts: list[str] = []
    for m in re.finditer(rb"stream[\r\n]+([\s\S]*?)[\r\n]+endstream", raw_bytes):
        data = m.group(1)
        try:
            decomp = zlib.decompress(data)
        except Exception:
            decomp = data
        # 1. Parenthesized literal strings
        for lit in re.finditer(rb"\(([^()\\]{2,400})\)", decomp):
            txt = lit.group(1).decode("utf-8", errors="ignore").strip()
            if txt and any(c.isalnum() for c in txt):
                parts.append(txt)
        # 2. UTF-16BE hex strings (common for Arabic Unicode PDFs)
        for hx in re.finditer(rb"<([0-9A-Fa-f]{4,800})>", decomp):
            hex_bytes = bytes.fromhex(hx.group(1).decode("ascii")) if len(hx.group(1)) % 2 == 0 else b""
            if hex_bytes.startswith(b"\xfe\xff"):
                txt = hex_bytes[2:].decode("utf-16-be", errors="ignore").strip()
                if txt:
                    parts.append(txt)
    if not parts:
        for lit in re.finditer(rb"\(([^()\\]{3,400})\)", raw_bytes):
            txt = lit.group(1).decode("utf-8", errors="ignore").strip()
            if txt and any(c.isalnum() for c in txt):
                parts.append(txt)
    return " ".join(parts).strip(), {"parser": "stdlib_flate_utf16be_fallback"}


def _inspect_image_bytes(filename: str, raw_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Inspect image header metadata safely; run OCR only if PIL + pytesseract are genuinely available."""
    meta: dict[str, Any] = {"byte_size": len(raw_bytes), "ocr_used": False, "visual_understanding": False}
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(raw_bytes) >= 24:
        w, h = struct.unpack(">II", raw_bytes[16:24])
        meta.update({"format": "PNG", "width": w, "height": h})
    elif raw_bytes.startswith(b"\xff\xd8"):
        meta["format"] = "JPEG"
    elif raw_bytes.startswith(b"BM") and len(raw_bytes) >= 26:
        w, h = struct.unpack("<ii", raw_bytes[18:26])
        meta.update({"format": "BMP", "width": abs(w), "height": abs(h)})
    elif raw_bytes.startswith((b"GIF87a", b"GIF89a")) and len(raw_bytes) >= 10:
        w, h = struct.unpack("<HH", raw_bytes[6:10])
        meta.update({"format": "GIF", "width": w, "height": h})
    else:
        meta["format"] = Path(filename).suffix.lstrip(".").upper() or "UNKNOWN"

    caps = detect_ocr_capabilities()
    if caps["ocr_ready"]:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(raw_bytes))
        ocr_text = pytesseract.image_to_string(img, lang="eng+fra+ara").strip()
        meta["ocr_used"] = True
        return ocr_text, meta

    meta["ocr_fallback_reason"] = "optional PIL/pytesseract OCR backend not installed; metadata extracted only"
    return f"[Image metadata only — OCR unavailable: {filename}, format={meta.get('format')}, size={len(raw_bytes)} bytes]", meta


def chunk_document(doc_id: str, filename: str, text: str, *, chunk_size: int = 360, overlap: int = 60) -> list[dict[str, Any]]:
    clean = (text or "").strip()
    if not clean:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    n = len(clean)
    step = max(50, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        snippet = clean[start:end].strip()
        if snippet:
            idx += 1
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{idx}",
                    "citation": f"[doc:{filename}#chunk_{idx}]",
                    "char_start": start,
                    "char_end": end,
                    "text": snippet,
                }
            )
        if end >= n:
            break
        start += step
    return chunks


class DocumentIntelligenceEngine:
    """SQLite-persisted multimodal document parser, table extractor, chunker, and citation-aware retriever."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_documents (
                    doc_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    language TEXT NOT NULL,
                    tables_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_doc_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    citation TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )

    def parse_bytes(self, filename: str, raw_bytes: bytes) -> ParsedDocument:
        if len(raw_bytes) > MAX_DOC_BYTES:
            raise ValueError(f"document exceeds maximum size ({MAX_DOC_BYTES} bytes)")
        ext = Path(filename or "doc.txt").suffix.lower()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        doc_id = f"doc_{sha[:10]}_{secrets.token_hex(3)}"
        tables: list[dict[str, Any]] = []
        meta: dict[str, Any] = {"extension": ext, "byte_size": len(raw_bytes)}

        if ext in (".txt", ".log", ""):
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            doc_type = "txt"
        elif ext in (".md", ".markdown"):
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            tables = _extract_markdown_tables(text)
            doc_type = "markdown"
        elif ext == ".json":
            raw_txt = raw_bytes.decode("utf-8", errors="replace")
            obj = json.loads(raw_txt)
            if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
                headers = sorted({k for r in obj for k in r.keys()})
                rows = [[str(r.get(h, "")) for h in headers] for r in obj[:200]]
                tables.append({"source": "json", "headers": headers, "rows": rows})
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            doc_type = "json"
        elif ext == ".csv":
            raw_txt = raw_bytes.decode("utf-8", errors="replace")
            reader = list(csv.reader(io.StringIO(raw_txt)))
            if reader:
                tables.append({"source": "csv", "headers": reader[0], "rows": reader[1:201]})
            text = "\n".join(" | ".join(r) for r in reader[:201])
            doc_type = "csv"
        elif ext == ".docx":
            text, tables = _extract_docx_text_and_tables(raw_bytes)
            doc_type = "docx"
        elif ext == ".pdf":
            text, pdf_meta = _extract_pdf_text(raw_bytes)
            meta.update(pdf_meta)
            doc_type = "pdf"
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            text, img_meta = _inspect_image_bytes(filename, raw_bytes)
            meta.update(img_meta)
            doc_type = "image"
        else:
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            doc_type = "text_fallback"

        lang = detect_text_language(text) if text else "english"
        meta["tables_count"] = len(tables)
        chunks = chunk_document(doc_id, filename, text)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            sha256=sha,
            language=lang,
            text=text,
            tables=tables,
            metadata=meta,
            chunks=chunks,
        )

    def ingest_bytes(self, user_id: str, filename: str, raw_bytes: bytes) -> ParsedDocument:
        parsed = self.parse_bytes(filename, raw_bytes)
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO v32_documents(
                    doc_id, user_id, filename, doc_type, sha256, language, tables_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parsed.doc_id,
                    user_id,
                    parsed.filename,
                    parsed.doc_type,
                    parsed.sha256,
                    parsed.language,
                    json.dumps(parsed.tables, ensure_ascii=False),
                    json.dumps(parsed.metadata, ensure_ascii=False),
                    now,
                ),
            )
            for ch in parsed.chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO v32_doc_chunks(
                        chunk_id, doc_id, user_id, filename, citation, char_start, char_end, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ch["chunk_id"],
                        parsed.doc_id,
                        user_id,
                        parsed.filename,
                        ch["citation"],
                        int(ch["char_start"]),
                        int(ch["char_end"]),
                        ch["text"],
                    ),
                )
        return parsed

    def ingest_file(self, user_id: str, path: str | Path) -> ParsedDocument:
        p = Path(path)
        return self.ingest_bytes(user_id, p.name, p.read_bytes())

    def list_documents(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT doc_id, filename, doc_type, sha256, language, tables_json, metadata_json, created_at
                FROM v32_documents WHERE user_id = ? ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                "doc_id": r[0],
                "filename": r[1],
                "doc_type": r[2],
                "sha256": r[3],
                "language": r[4],
                "tables": json.loads(r[5]),
                "metadata": json.loads(r[6]),
                "created_at": float(r[7]),
            }
            for r in rows
        ]

    def retrieve_with_citations(self, user_id: str, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Citation-aware TF-IDF cosine retrieval over ingested multimodal document chunks."""
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, filename, citation, char_start, char_end, text
                FROM v32_doc_chunks WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()
        if not rows:
            return []
        docs_toks = [_tokenize(r[6]) for r in rows]
        n_docs = len(docs_toks)
        df: Counter[str] = Counter()
        for toks in docs_toks:
            for t in set(toks):
                df[t] += 1

        def vec(toks: list[str]) -> dict[str, float]:
            tf = Counter(toks)
            tot = max(1, len(toks))
            return {t: (c / tot) * (math.log((n_docs + 1.0) / (df.get(t, 0) + 1.0)) + 1.0) for t, c in tf.items()}

        qv = vec(q_toks)
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scored: list[tuple[float, dict[str, Any]]] = []
        for r, dtoks in zip(rows, docs_toks):
            dv = vec(dtoks)
            dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            cos = sum(qv[t] * dv.get(t, 0.0) for t in qv) / (qn * dn)
            if cos > 0.0:
                scored.append(
                    (
                        cos,
                        {
                            "chunk_id": r[0],
                            "doc_id": r[1],
                            "filename": r[2],
                            "citation": r[3],
                            "char_span": [int(r[4]), int(r[5])],
                            "text": r[6],
                            "score": round(cos, 4),
                            "reliability": 0.95,
                        },
                    )
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: max(1, int(top_k))]]
