# نتائج اختبارات المتصفح والنهاية-إلى-النهاية — مشروع NourAI (`BROWSER_E2E_RESULTS.md`)

- **تاريخ التنفيذ:** 2026-11-17

---

## 1. اختبارات Playwright في متصفح حقيقي (`Chromium`)
- **الملف المُنشأ:** [tests/e2e_playwright_browser.py](file:///d:/NourAI/tests/e2e_playwright_browser.py)
- **التكامل المستمر:** تمت إضافة `playwright` + `playwright install chromium` إلى [.github/workflows/ci.yml](file:///d:/NourAI/.github/workflows/ci.yml).
- **الحالة المحلية:** **لم يُشغّل محلياً** (يُتخطى عبر `pytest.importorskip("playwright.sync_api")`) لأن `playwright` غير مثبت في `.venv`.

---

## 2. اختبارات E2E الحية + تنفيذ JavaScript الفعلي (`Node.js` + `HTTP`)
- **الملف المنفذ:** [tests/test_e2e_ui_and_concurrency.py](file:///d:/NourAI/tests/test_e2e_ui_and_concurrency.py) (`4 passed in 4.71s`, وضمن الحزمة الكاملة `72 passed in 25.26s`).
- **ما تم التحقق منه فعلياً:**
  1. جلب `GET /` من الخادم الحي والتحقق من ترويسة `Content-Security-Policy` الصارمة (`script-src 'self' 'sha256-...'` بدون `'unsafe-inline'`, `object-src 'none'`, `frame-ancestors 'none'`).
  2. دورة مستخدم كاملة عبر HTTP: تسجيل -> توكن -> محادثة -> حذف الحساب.
  3. استخراج دوال التعقيم الفعلية من [ui/index.html](file:///d:/NourAI/ui/index.html) وتشغيلها في `Node.js` ضد حمولات XSS (`<svg onload=...>`, كسر سمات Markdown `"onerror=`, وكسر `'` في كتل الكود).
  4. عزل إيقاف التوليد بين المستخدمين (`test_cq01_per_user_stop_isolation_and_concurrency`) ورفض معاملات التوليد غير الصالحة برمز `400` (`test_cq02_invalid_generation_params_return_400`).
