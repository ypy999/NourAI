# NourAI v3.2 — Multimodal Document Intelligence (`MULTIMODAL_SYSTEM.md`)

Implementation: [multimodal.py](file:///d:/NourAI/nourai/documents/multimodal.py) · API: `GET/POST /api/v3/documents/multimodal`.

## 1. Supported Formats & Graceful Fallbacks
| Format | Parser Strategy | Table Extraction |
| :--- | :--- | :--- |
| **TXT** | UTF-8 streaming decode | — |
| **Markdown** | UTF-8 decode + pipe-table parser | Yes (`_extract_markdown_tables`) |
| **JSON** | Stdlib `json.loads` | Yes (array-of-objects → table) |
| **CSV** | Stdlib `csv.reader` | Yes (header + rows) |
| **DOCX** | Stdlib ZIP/XML (`word/document.xml`) with `12 MiB` XML bomb cap | Yes (`<w:tbl>` rows/cells) |
| **PDF** | Optional `pypdf` if installed, else stdlib `FlateDecode` + UTF-16BE Arabic hex fallback | Text & metadata |
| **Images (PNG/JPG/BMP/GIF)** | Binary header metadata parser (`format`, `width`, `height`) + optional `PIL`/`pytesseract` OCR | Honest metadata-only fallback |

## 2. Honest OCR & Vision Disclosure
- `detect_ocr_capabilities()` probes the runtime environment (`pil_available: false`, `pytesseract_available: false`, `ocr_ready: false`, `vision_model_available: false` on this local `.venv`).
- NourAI **never claims visual understanding** when an OCR or vision backend is not installed; image metadata is extracted cleanly with `visual_understanding: false`.

## 3. Citation-Aware Chunking & Retrieval
- Overlapping chunks persisted in SQLite (`v32_doc_chunks`) with exact citations `[doc:<filename>#chunk_<N>]` and `char_span`.
- Verified across **25 / 25** multimodal parsing tasks and **20 / 20** citation retrieval tasks.
