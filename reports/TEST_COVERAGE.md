# تقرير تغطية وجودة الاختبارات — مشروع NourAI (`TEST_COVERAGE.md`)

- **نطاق الفحص:** مراجعة ملفات الاختبارات في `tests/` (51 ملف اختبار) وتشغيل الاختبارات غير التدميرية في البيئة الافتراضية `.venv`.
- **تاريخ التدقيق:** 2026-11-17

---

## 1. نتائج التشغيل الفعلي للاختبارات

تم تشغيل حزمة الاختبارات الأساسية (10 ملفات تشمل العمارة، التدريب، الاستدلال، API، UI، الأمان والأداء، الإنتاج والإصدارات، المحركات الثمانية، والتقييم) باستخدام:
```powershell
d:\NourAI\.venv\Scripts\python.exe -m pytest tests/test_phase11_architecture.py tests/test_phase12_training_engine.py tests/test_phase17_inference_engine.py tests/test_phase18_api_backend.py tests/test_phase19_user_interface.py tests/test_phase20_security_performance.py tests/test_phase21_to_24_production_and_release.py tests/test_nourai_8_engines.py tests/test_training.py tests/test_evaluation.py -v --tb=short
```
- **النتيجة الملاحظة:** **`58 passed in 25.83s`** (جميع الحالات الـ 58 نجحت).
- **ملاحظة بيئية:** عند التشغيل الافتراضي بدون تجاوز `cache_dir`، فشل `pytest` في الكتابة إلى `D:\NourAI\.pytest_cache` بسبب قيود صلاحيات على ذلك المجلد (`PermissionDenied`).

---

## 2. نواقص ومشاكل الاختبارات المكتشفة

### `TST-01` — اختبارات القبول تقرأ تقارير مخزنة مسبقاً ولا تنفّذ الفحص الفعلي (`Stale Report Caching`)
- **الخطورة:** `High`
- **الملفات والأسطر:**
  - `tests/test_phase18_api_backend.py:11-17`
  - `tests/test_phase19_user_interface.py:11-17`
  - `tests/test_phase20_security_performance.py:11-17`
- **الدليل:**
  ```python
  if not report_path.is_file():
      report = run_phase20_security_and_performance_audit()
      write_phase20_reports(report, "reports/phase20")
  payload = json.loads(report_path.read_text(encoding="utf-8"))
  ```
  بما أن ملفات التقارير موجودة مسبقاً على القرص، فإن هذه الاختبارات تتخطى استدعاء دوال التدقيق وتقرأ الملف القديم، مما يجعلها تنجح حتى لو تعطل الكود الفعلي!
- **الحل العملي:** تشغيل دوال التدقيق مباشرة في كل اختبار داخل مجلد مؤقت (`tmp_path`) دون الاعتماد على وجود ملفات تقارير قديمة.

### `TST-02` — غياب اختبارات سلبية للثغرات الحرجة والعالية
- **الخطورة:** `High`
- **التفاصيل:** لا توجد أي اختبارات تتحقق من:
  1. منع تسجيل مستخدم جديد بصلاحية `admin` عبر `POST /api/v1/users/register` (`SEC-01`).
  2. منع تجاوز قائمة الحظر في `POST /api/v1/code/run` (`SEC-02`).
  3. حظر XSS في `ui/index.html` عند وجود `<svg onload=...>` أو `'` في الكود (`SEC-03`, `SEC-04`).
  4. منع SSRF في `_maybe_read_and_summarize_url` ضد `127.0.0.1` و`169.254.169.254` (`SEC-05`).
  5. تطبيق حد حجم جسم الطلب (`413 Payload Too Large`) ومعالجة JSON التالف في `deployment/production.py` (`SEC-07`, `CQ-02`).
  6. تطبيق `Rate Limiting` على `/api/v1/auth/token` و`/api/v1/users/register` (`SEC-09`).
  7. حذف بيانات RAG عند حذف المستخدم `DELETE /api/v1/users/me` (`SEC-16`).

### `TST-03` — غياب اختبارات نهاية-إلى-نهاية (`E2E Browser Tests`) لواجهة المستخدم
- **الخطورة:** `Medium`
- **التفاصيل:** يكتفي `tests/test_phase19_user_interface.py` بفحص وجود سلاسل نصية داخل `ui/index.html` (`'dir="rtl"' in html`)، ولا يختبر تنفيذ JavaScript الفعلي أو معالجة Markdown/SVG/Quiz/Workflow في متصفح حقيقي.
- **الحل العملي:** إضافة اختبارات Playwright خفيفة للتحقق من تدفق المحادثة وتجارب XSS في المتصفح.

### `TST-04` — غياب سير عمل CI آلي في `.github/workflows/`
- **الخطورة:** `Medium`
- **التفاصيل:** يحتوي `.github/workflows/` على نشر Firebase Hosting فقط، ولا يوجد workflow لتشغيل `pytest` تلقائياً عند التعديلات.
