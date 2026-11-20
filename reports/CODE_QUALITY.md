# تقرير جودة الكود والمعمارية والأداء — مشروع NourAI (`CODE_QUALITY.md`)

- **نطاق الفحص:** البنية المعمارية، معالجة الأخطاء، كفاءة قاعدة البيانات، الأداء والتزامن، والتناسق البرمجي عبر وحدات `api/`, `deployment/`, `nourai/`, `inference/`, `model/`, `training/`.
- **تاريخ التدقيق:** 2026-11-17

---

## ملخص النتائج

| المعرف | الخطورة | التصنيف | الملف والأسطر |
|---|---|---|---|
| `CQ-01` | **High** | تزامن وحالة مشتركة عالمياً (`Global Stop Event`) | `api/backend.py:749-750`, `inference/engine.py:122-130` |
| `CQ-02` | **High** | معالجة استثناءات مفقودة (HTTP 500 غير متوقع) | `api/backend.py:665, 794-799`, `deployment/production.py:251-253` |
| `CQ-03` | **Medium** | كفاءة SQLite وفتح اتصالات متكررة بدون Pooling | `api/backend.py:43-46, 225-228`, `nourai/memory_and_rag.py:106-109` |
| `CQ-04` | **Medium** | تحميل كامل قاعدة البيانات في الذاكرة عند الإقلاع + غياب الفهارس | `api/backend.py:65-80, 256-278`, `nourai/memory_and_rag.py:302-320` |
| `CQ-05` | **Medium** | تصادم معرفات المستندات (`doc_id`) عند التشابه في الاسم والطول | `nourai/memory_and_rag.py:240` |
| `CQ-06` | **Medium** | أداء الاستدلال: إعادة تمرير السياق كاملاً بدون KV-Cache | `inference/engine.py:199-202` |
| `CQ-07` | **Medium** | تضخم ملفات منفردة (`God Modules`) وصعوبة الصيانة | `api/backend.py` (1050 سطر), `nourai/web_search.py` (1033 سطر), `ui/index.html` (3074 سطر) |
| `CQ-08` | **Low** | خلط كود الإنتاج مع دوال التدقيق المرحلية (`Phase Audits`) | `api/backend.py:936-1050`, `inference/engine.py:294-502`, `deployment/production.py:311-546` |
| `CQ-09` | **Low** | تخزين غير متوافق لقيم الجلسات بين الذاكرة وSQLite (`max_turns_per_session`) | `api/backend.py:365-373` |

---

## تفاصيل الملاحظات والحلول

### `CQ-01` — حالة إيقاف التوليد المشتركة عالمياً (`Global _stop_event`)
- **الخطورة:** `High`
- **الموقع:** `api/backend.py` (الأسطر `749-750`, `803`), `inference/engine.py` (الأسطر `122-130`)
- **التفاصيل:** يستدعي المسار `POST /api/v1/chat/stop` الدالة `self.engine.request_stop()` التي تضبط `threading.Event` مشتركاً على مستوى كائن `InferenceEngine` الوحيد في الخادم.
- **الأثر:** إذا أرسل مستخدم طلب إيقاف، فسيتم إلغاء عمليات التوليد الجارية لجميع المستخدمين الآخرين، وإذا بدأ طلب جديد `reset_stop()` فسيُلغي إشارة الإيقاف لطلب آخر.
- **الحل:** ربط `cancel_event: threading.Event` بكل جلسة أو معرف طلب (`request_id`) وتمريره مباشرة إلى `stream_generate(..., cancel_event=...)` الموجود أصلاً في `InferenceEngine`.

### `CQ-02` — معالجة استثناءات التحويل المفقودة في واجهة API والخادم
- **الخطورة:** `High`
- **الموقع:** `api/backend.py` (الأسطر `665`, `794-799`), `deployment/production.py` (الأسطر `251-253`)
- **التفاصيل:**
  1. `base64.b64decode(b64_content)` في `POST /api/v1/documents` لا يعالج `binascii.Error`.
  2. `int(body.get("max_new_tokens", 160))` و`float(body.get("temperature", 0.7))` في `POST /api/v1/chat` لا تعالج `ValueError` أو `TypeError` (كما أن استثناءات `GenerationConfig.validate()` من نوع `ValueError` غير محاطة بـ `try/except` لإرجاع `400`).
  3. `deployment/production.py` لا يلتقط `json.JSONDecodeError` أو `UnicodeDecodeError`.
- **الحل:** تغليف قراءة وتحليل المدخلات بكتل `try/except` موحدة ترجع استجابة JSON منظمة برمز `400 Bad Request`.

### `CQ-03` — تكرار تهيئة اتصالات SQLite وتفعيل WAL في كل عملية
- **الخطورة:** `Medium`
- **الموقع:** `api/backend.py` (الأسطر `43-46`, `225-228`), `nourai/memory_and_rag.py` (الأسطر `106-109`, `182-185`)
- **التفاصيل:** كل استدعاء لـ `_connect()` يفتح اتصالاً جديداً ويعيد تنفيذ `PRAGMA journal_mode=WAL`؛ ومع وجود 4 مدراء منفصلين (`UserManager`, `SessionManager`, `LongTermMemoryStore`, `LocalDocumentRAGStore`) على نفس ملف قاعدة البيانات، يزداد خطر تنافس الأقفال (`SQLITE_BUSY`) تحت الحمل المتزامن.
- **الحل:** توحيد طبقة الوصول لقاعدة البيانات في كائن واحد يدير التهيئة، مع تطبيق `PRAGMA journal_mode=WAL` مرة واحدة عند التهيئة.

### `CQ-04` — تحميل كافة المستخدمين والجلسات والرسائل في الذاكرة عند بدء التشغيل
- **الخطورة:** `Medium`
- **الموقع:** `api/backend.py` (الأسطر `65-80`, `256-278`), `nourai/memory_and_rag.py` (الأسطر `302-320`)
- **التفاصيل:**
  - الدالتان `_load_from_db()` تقومان بتحميل جميع صفوف `prod_users` و`prod_sessions` و`prod_messages` في قواميس الذاكرة (مع استعلام N+1 لكل جلسة في `SessionManager._load_from_db` الأسطر `263-267`).
  - `LocalDocumentRAGStore.search_chunks` يقرأ **جميع** مقاطع المستندات الخاصة بالمستخدم من SQLite ثم يفرزها في Python بدون فهرسة (`FTS5`)، كما تخلو الجداول من فهارس على `user_id` و`session_id`.
- **الحل:** الاستعلام عند الحاجة من SQLite مع إضافة فهارس (`CREATE INDEX IF NOT EXISTS`) على `prod_sessions(user_id)`، `prod_messages(session_id)`، `prod_user_memories(user_id)`، و`prod_rag_chunks(user_id)`.

### `CQ-05` — تصادم معرفات مستندات RAG (`doc_id`)
- **الخطورة:** `Medium`
- **الموقع:** `nourai/memory_and_rag.py` (السطر `240`)
- **التفاصيل:** يُحسب `doc_id` من `hashlib.sha256(f"{user_id}:{clean_name}:{len(clean_text)}"...)[:16]` فقط دون تضمين محتوى النص. رفع ملفين بنفس الاسم ونفس عدد الأحرف بمحتوى مختلف يستبدل الأول.
- **الحل:** تضمين `hashlib.sha256(clean_text.encode("utf-8")).hexdigest()` في حساب `doc_id`.

### `CQ-06` — غياب KV-Cache في `InferenceEngine.stream_generate`
- **الخطورة:** `Medium`
- **الموقع:** `inference/engine.py` (الأسطر `199-202`)
- **التفاصيل:** في كل خطوة توليد، يعيد المحرك تمرير كامل نافذة السياق `(context_ids + generated_ids)[-max_ctx:]` عبر جميع طبقات المحول $O(T^2)$ بدلاً من تخزين Key/Value Cache.
- **الأثر:** مقبول حالياً لنموذج 843K و`context_length=128`، لكنه سيسبب بطئاً حاداً عند التوسع إلى نماذج `small` أو `base` أو سياق أطول.

### `CQ-07` — ملفات ضخمة متعددة المسؤوليات (`God Modules`)
- **الخطورة:** `Medium`
- **الموقع:** `api/backend.py` (1050 سطر)، `nourai/web_search.py` (1033 سطر)، `ui/index.html` (3074 سطر)
- **التفاصيل:** تدمج هذه الملفات إدارة المستخدمين، المصادقة، التوجيه، RAG، تكامل LLM، HTML/CSS/JS، ومنطق التحقق المرحلي في ملف واحد.
- **الحل:** تقسيم `api/backend.py` إلى `auth.py`, `users.py`, `sessions.py`, `routes.py`، وفصل JS/CSS من `ui/index.html` إلى ملفات مستقلة (مما يتيح أيضاً إزالة `'unsafe-inline'` من CSP).

### `CQ-08` — اختلاط كود الإنتاج بدوال التدقيق المرحلي (`run_phase*` / `run_stage_*`)
- **الخطورة:** `Low`
- **الموقع:** `api/backend.py:936-1050`, `inference/engine.py:294-502`, `deployment/production.py:311-546`, `nourai/capabilities.py:348-523`
- **التفاصيل:** تحتوي وحدات الإنتاج على دوال تدقيق وتقارير طويلة تكتب ملفات تحت `reports/`، مما يزيد حجم الكود التشغيلي ويخلط مسؤوليات التشغيل بالتحقق.
- **الحل:** نقل دوال `run_phase*` و`run_stage_*` إلى حزمة مخصصة للتدقيق أو ضمن `tests/` / `scripts/`.

### `CQ-09` — عدم اتساق تقليم الرسائل بين الذاكرة وSQLite
- **الخطورة:** `Low`
- **الموقع:** `api/backend.py` (الأسطر `365-373`)
- **التفاصيل:** في `SessionManager.append_turn`، يتم تقليم `sess["messages"]` في الذاكرة إلى `max_turns_per_session * 2`، بينما يستمر الإدراج في جدول `prod_messages` بلا حذف، وعند إعادة التحميل `_load_from_db` تُحمّل جميع الرسائل القديمة دون تطبيق الحد.
- **الحل:** تطبيق نفس حد الرسائل في استعلام التحميل أو تنظيف الرسائل الأقدم دورياً.
