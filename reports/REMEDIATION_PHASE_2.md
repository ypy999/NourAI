# تقرير معالجة المرحلة الثانية (High Security Remediation) — `REMEDIATION_PHASE_2.md`

- **تاريخ التنفيذ:** 2026-11-17
- **النطاق:** إغلاق الثغرات عالية الخطورة `SEC-03`, `SEC-04`, `SEC-05`, `SEC-06` دون التراجع عن إصلاحات المرحلة الأولى ودون البدء بالمرحلة الثالثة.

---

## 1. الثغرات التي تم إغلاقها

### `SEC-03` & `SEC-04` — إغلاق ثغرات XSS وتأمين معالجة DOM والأحداث في `ui/index.html`
- **الملف المعدل:** [ui/index.html](file:///d:/NourAI/ui/index.html)
- **التفاصيل:**
  1. تشديد `escapeHtml(str)` ليشمل ترميز علامات الاقتباس المزدوجة والمفردة (`"` -> `&quot;`, `'` -> `&#39;`) لمنع كسر السمات في روابط وصور Markdown.
  2. إزالة الحقن المباشر لوسوم `<svg>` الخام داخل `innerHTML`؛ واستبداله بدالة `sanitizeSvgForImg` وعرض الرسم عبر `<img src="data:image/svg+xml;charset=utf-8,...">` الذي يمنع المتصفح من تنفيذ أي سكربتات أو أحداث داخل SVG.
  3. استبدال جميع معالجات `onclick` الديناميكية المبنية عبر `decodeURIComponent(...)` في `renderQuizBlock` و`renderKGraphBlock` و`renderWorkflowBlock` و`renderFormattedContent` بآلية تفويض أحداث موحدة (`Event Delegation` عبر `data-ui-action` و`_uiActionPayloads` `Map`).
  4. استبدال `stepEl.innerHTML += ...` في `runNextWorkflowStep` بإنشاء عنصر DOM (`document.createElement("div")`) واستخدام `textContent`.
- **الاختبار الأمني:** `test_sec03_sec04_ui_xss_hardening` في [tests/test_phase19_user_interface.py](file:///d:/NourAI/tests/test_phase19_user_interface.py).

### `SEC-05` — حماية `nourai/web_search.py` من SSRF وDNS Rebinding والتحويلات
- **الملف المعدل:** [nourai/web_search.py](file:///d:/NourAI/nourai/web_search.py)
- **التفاصيل:**
  1. إضافة `validate_public_url()` التي تقتصر على `http`/`https`، وترفض بيانات الاعتماد في الرابط، وتحظر `localhost` و`.localhost` و`.local` و`.internal` وعناوين metadata (`metadata.google.internal`, `169.254.169.254`)، وتفحص جميع عناوين IP المحلولة عبر `socket.getaddrinfo` للتأكد من كونها عناوين عامة (`is_global` وليست `private`/`loopback`/`link_local`/`multicast`/`reserved`/`unspecified`).
  2. إضافة `_PinnedHTTPConnection` و`_PinnedHTTPSConnection` و`_safe_fetch_public_url()` للاتصال مباشرة بعنوان IP العام الذي تم التحقق منه مسبقاً (لمنع هجمات `DNS Rebinding` / TOCTOU) مع الحفاظ على `Host` و`TLS SNI`.
  3. التحقق من كل وجهة إعادة توجيه (`301/302/303/307/308`) بحد أقصى 3 تحويلات، وتحديد حد أقصى لحجم الاستجابة (`1,000,000` بايت).
  4. تحديث `_maybe_read_and_summarize_url` لاستخدام `_safe_fetch_public_url`.
- **الاختبار الأمني:** `test_sec05_ssrf_url_validation_and_redirect_blocking` في [tests/test_phase18_api_backend.py](file:///d:/NourAI/tests/test_phase18_api_backend.py).

### `SEC-06` — منع إرسال الذكريات ومقاطع RAG إلى الخدمات الخارجية بدون موافقة صريحة
- **الملفات المعدلة:** [nourai/web_search.py](file:///d:/NourAI/nourai/web_search.py)، [api/backend.py](file:///d:/NourAI/api/backend.py)، [ui/index.html](file:///d:/NourAI/ui/index.html)
- **التفاصيل:**
  1. إضافة معامل `allow_cloud_context: bool = False` في `NourAIWebAndSmartEngine.answer_with_web` و`query_free_llm_with_context`.
  2. افتراضياً (`allow_cloud_context=False`)، لا تُرسل `user_memories` أو `rag_chunks` أو `history` إلى `text.pollinations.ai` أو `generativelanguage.googleapis.com`، مع بقاء RAG المحلي (`local_document_rag`) يعمل محلياً بالكامل.
  3. نقل مفتاح Gemini من معامل URL (`?key=...`) إلى الترويسة `x-goog-api-key`.
  4. إضافة خيار موافقة صريحة غير مفعل افتراضياً (`id="cloud-context-consent-toggle"`) في إعدادات [ui/index.html](file:///d:/NourAI/ui/index.html) وتمرير `allow_cloud_context` عبر [api/backend.py](file:///d:/NourAI/api/backend.py).
- **الاختبار الأمني:** `test_sec06_cloud_context_requires_explicit_consent` في [tests/test_phase18_api_backend.py](file:///d:/NourAI/tests/test_phase18_api_backend.py).

---

## 2. نتائج الاختبارات بعد الإصلاح
- **حزمة `pytest`** (`test_phase18_api_backend.py`, `test_phase19_user_interface.py`, `test_phase20_security_performance.py`, `test_phase21_to_24_production_and_release.py`, `test_nourai_8_engines.py`, `test_evaluation.py`):
  - **`33 passed in 12.41s`** (شاملة اختبارات المرحلة الأولى والثانية).
- **تدقيق المرحلة 20 الفعلي (`run_phase20_security_and_performance_audit()`):**
  - **`CERTIFIED_FOR_PHASE_21_DEPLOYMENT_AND_PRODUCTION`** (نجاح جميع البوابات السبع).

---

## 3. الثغرات الأمنية المتبقية (للمراحل القادمة)
- **High:** `SEC-07` (حد حجم `Content-Length` في `deployment/production.py`)، `SEC-08` (تمرير `client_ip` الفعلي)، `SEC-09` (`Rate Limiting` على المصادقة والتسجيل).
- **Medium:** `SEC-10` إلى `SEC-15` (علامات الكوكيز، تخزين مفتاح Gemini في الكوكيز، فحص أصل CORS، معامل `?api=`، Zip Bomb في `.docx`، حقن الموجهات غير المباشر).
- **Low:** `SEC-16` إلى `SEC-18` (تنظيف RAG عند حذف المستخدم، صلاحيات الملفات/HSTS، إزالة fallback غير الآمن في `torch.load`).
