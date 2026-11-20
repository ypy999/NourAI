# تقرير معالجة المرحلة الرابعة والأخيرة (Medium & Low Security Remediation) — `REMEDIATION_PHASE_4.md`

- **تاريخ التنفيذ:** 2026-11-17
- **النطاق:** معالجة الثغرات المتوسطة (`SEC-10`–`SEC-15`) والمنخفضة (`SEC-16`–`SEC-18`)، ومراجعة إعدادات Docker وFirebase والإنتاج.

---

## 1. الثغرات التي تمت معالجتها في المرحلة الرابعة

| المعرف | الخطورة | الملفات المعدلة | ملخص الإصلاح | الاختبار المثبت |
|---|---|---|---|---|
| `SEC-10` | **Medium** | [api/backend.py](file:///d:/NourAI/api/backend.py) | إضافة `HttpOnly` ومواءمة `Max-Age={ttl_seconds}` (3600) وإضافة `Secure` عند HTTPS في `Set-Cookie`. | `test_sec10_sec12_sec16_cookies_cors_and_user_deletion_rag_cleanup` |
| `SEC-11` | **Medium** | [ui/index.html](file:///d:/NourAI/ui/index.html) | نقل `nourai_gemini_key` من `document.cookie` إلى `sessionStorage` ومسح الكوكي القديم تلقائياً. | `test_sec03_sec04_ui_xss_hardening` |
| `SEC-12` | **Medium** | [api/backend.py](file:///d:/NourAI/api/backend.py) | استبدال `startswith("http://localhost")` بفحص `urlparse` دقيق للـ `hostname` وإضافة `Vary: Origin`. | `test_sec10_sec12_sec16_cookies_cors_and_user_deletion_rag_cleanup` |
| `SEC-13` | **Medium** | [ui/index.html](file:///d:/NourAI/ui/index.html) | تقييد `isSafeApiUrl` بقائمة بيضاء `_TRUSTED_API_ORIGINS` + `localhost`/`127.0.0.1` وإيقاف الحفظ التلقائي من `?api=` في `localStorage`. | `test_sec03_sec04_ui_xss_hardening` |
| `SEC-14` | **Medium** | [nourai/memory_and_rag.py](file:///d:/NourAI/nourai/memory_and_rag.py), [api/backend.py](file:///d:/NourAI/api/backend.py) | فرض حد `5 MiB` للملف و`10 MiB` لـ `word/document.xml` داخل `.docx` (منع Zip Bomb)، ومعالجة `binascii.Error` برمز `400`. | `test_sec14_sec15_sec17_sec18_upload_limits_prompt_isolation_hsts_and_torch_load` |
| `SEC-15` | **Medium** | [nourai/web_search.py](file:///d:/NourAI/nourai/web_search.py), [nourai/engines.py](file:///d:/NourAI/nourai/engines.py) | عزل البيانات الخارجية داخل `<external_data>` مع تعليمات تجاهل الأوامر المضمنة؛ تقييد `ast.Pow` في `_safe_eval_ast`. | `test_sec14_sec15_sec17_sec18_upload_limits_prompt_isolation_hsts_and_torch_load` |
| `SEC-16` | **Low** | [api/backend.py](file:///d:/NourAI/api/backend.py), [nourai/memory_and_rag.py](file:///d:/NourAI/nourai/memory_and_rag.py) | جعل `user_id` عشوائياً (`usr_{secrets.token_hex(8)}`) وإضافة `clear_user_documents` وتنفيذها عند `DELETE /api/v1/users/me`. | `test_sec10_sec12_sec16_cookies_cors_and_user_deletion_rag_cleanup` |
| `SEC-17` | **Low** | [run_nourai.py](file:///d:/NourAI/run_nourai.py), [deployment/production.py](file:///d:/NourAI/deployment/production.py), [Dockerfile](file:///d:/NourAI/Dockerfile) | تقييد صلاحيات `.jwt_secret` و`localhost_key.pem` بـ `0o600`، إرسال HSTS فقط عند `enable_tls=True`، وتشغيل `Dockerfile` بمستخدم غير جذري `appuser` ومجلد بيانات `0700`. | `test_sec14_sec15_sec17_sec18_upload_limits_prompt_isolation_hsts_and_torch_load` |
| `SEC-18` | **Low** | [model/transformer.py](file:///d:/NourAI/model/transformer.py) | إزالة مسار fallback غير الآمن في `GPT.load_checkpoint` والاعتماد حصرياً على `weights_only=True`. | `test_sec14_sec15_sec17_sec18_upload_limits_prompt_isolation_hsts_and_torch_load` |

---

## 2. مراجعة Docker وFirebase والإنتاج
- **`Dockerfile`:** تم تحويل التشغيل من `root` و`chmod -R 755 /app` إلى مستخدم غير جذري `appuser` (`uid 10001`) مع `chmod 700 /app/data/production`.
- **`firebase.json`:** يتضمن ترويسات أمنية مناسبة (`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `Permissions-Policy`) ويستبعد ملفات `.py` والملفات المخفية من النشر الساكن.
- **`render.yaml`:** يوصى بإضافة متغيرات البيئة `NOURAI_SECRET_KEY` و`NOURAI_TRUSTED_PROXIES` في لوحة Render قبل النشر.
