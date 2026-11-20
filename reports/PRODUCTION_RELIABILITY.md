# تقرير موثوقية الإنتاج — مشروع NourAI (`PRODUCTION_RELIABILITY.md`)

- **تاريخ التنفيذ:** 2026-11-17
- **النطاق:** المراحل A و B و C و D لنقل NourAI نحو الاستقرار والموثوقية والجاهزية للنشر.

---

## 1. المرحلة A — التزامن وعزل الإيقاف والتحقق من المدخلات
- **`CQ-01` (عزل إيقاف التوليد حسب المستخدم والطلب):**
  - تم استبدال `self.engine.request_stop()` / `reset_stop()` العالمي في [api/backend.py](file:///d:/NourAI/api/backend.py) بآلية `_active_cancellations` محمية بـ `threading.Lock()` ومفهرسة بـ `(user_id, session_id)` لكل طلب.
  - عند استدعاء `POST /api/v1/chat/stop` يتم إلغاء طلبات المستخدم المصادق عليه فقط (أو الجلسة المحددة إن أُرسلت)، دون التأثير على أي مستخدم آخر.
- **`CQ-02` (التحقق من معاملات التوليد وإرجاع `400 Bad Request`):**
  - تغليف تحليل `max_new_tokens`, `strategy`, `temperature`, `top_k`, `top_p`, `repetition_penalty` (مع رفض القيم المنطقية `bool`) واستدعاء `GenerationConfig.validate()` بكتلة `try/except (ValueError, TypeError)` لإرجاع `400` منظمة.
- **الاختبارات:** `test_cq01_per_user_stop_isolation_and_concurrency` و`test_cq02_invalid_generation_params_return_400` في [tests/test_e2e_ui_and_concurrency.py](file:///d:/NourAI/tests/test_e2e_ui_and_concurrency.py).

---

## 2. المرحلة B — تحسين SQLite والفهارس وKV-Cache
- **تحسينات SQLite (`CQ-04`, `CQ-05`, `CQ-09`):**
  - إضافة الفهارس `idx_prod_sessions_user_id`, `idx_prod_messages_session_id`, `idx_prod_user_memories_user_id`, `idx_prod_rag_docs_user_id`, `idx_prod_rag_chunks_user_id`, `idx_prod_metrics_name_time` في [api/backend.py](file:///d:/NourAI/api/backend.py) و[nourai/memory_and_rag.py](file:///d:/NourAI/nourai/memory_and_rag.py) و[deployment/production.py](file:///d:/NourAI/deployment/production.py).
  - معالجة استعلام N+1 في `SessionManager._load_from_db()` باستعلام موحد وتطبيق حد `max_turns_per_session * 2` عند التحميل.
  - تضمين تجزئة المحتوى في `doc_id` بـ `LocalDocumentRAGStore.ingest_document` لمنع تصادم الملفات.
- **KV-Cache (`CQ-06`):**
  - إضافة `Block.forward_with_cache` و`GPT.forward_with_cache` في [model/transformer.py](file:///d:/NourAI/model/transformer.py) وتفعيله في `InferenceEngine.stream_generate` في [inference/engine.py](file:///d:/NourAI/inference/engine.py) مع التحقق من التطابق العددي التام مع المسار غير المخزن (`test_stage_b_sqlite_indexes_and_kv_cache_equivalence`).

---

## 3. المرحلة C — الاعتماديات والتكامل المستمر (CI)
- **ملف الاعتماديات المقيد:** إنشاء [requirements.lock](file:///d:/NourAI/requirements.lock) بإصدارات ثابتة (`==`) من البيئة الحالية. *(ملاحظة: لم يتم توليد `--require-hashes` لروابط PyTorch CPU/CUDA المتعددة الأنظمة لأن البيئة لا تحتوي على `pip-audit` أو `uv`، ولم يتوفر `pip-audit` محلياً للتشغيل المباشر).*
- **CI Workflow:** إنشاء [.github/workflows/ci.yml](file:///d:/NourAI/.github/workflows/ci.yml) على Python `3.11` (متوافق مع `Dockerfile` و`render.yaml`) لتشغيل `pip-audit` وكافة اختبارات `pytest`.

---

## 4. المرحلة D — اختبارات E2E وتشديد CSP ومراجعة Docker/TLS
- **اختبارات E2E وتنفيذ JS الفعلي:** إضافة `test_stage_d_e2e_live_http_and_ui_js_xss_defense` في [tests/test_e2e_ui_and_concurrency.py](file:///d:/NourAI/tests/test_e2e_ui_and_concurrency.py) التي تختبر دورة HTTP كاملة على الخادم الحي وتنفذ دوال التعقيم الفعلية من [ui/index.html](file:///d:/NourAI/ui/index.html) عبر `Node.js` ضد حمولات XSS.
- **تشديد CSP:** تحديث `Content-Security-Policy` في [api/backend.py](file:///d:/NourAI/api/backend.py) لإضافة `object-src 'none'`, `base-uri 'self'`, `frame-ancestors 'none'` مع بقاء `'unsafe-inline'` في `script-src`/`style-src` نظراً لوجود معالجات `onclick` ثابتة وسكربت مضمن في `ui/index.html`.
