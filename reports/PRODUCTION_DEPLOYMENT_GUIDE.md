# دليل النشر الإنتاجي — مشروع NourAI (`PRODUCTION_DEPLOYMENT_GUIDE.md`)

- **تاريخ التحديث:** 2026-11-17

---

## 1. متغيرات البيئة المطلوبة في الإنتاج
- `NOURAI_SECRET_KEY`: مفتاح سري عشوائي قوي (تمت إضافته في [render.yaml](file:///d:/NourAI/render.yaml) مع `generateValue: true`، وتحديث [run_nourai.py](file:///d:/NourAI/run_nourai.py) لقراءته قبل `.jwt_secret`).
- `NOURAI_TRUSTED_PROXIES`: قائمة عناوين/شبكات الوكيل الموثوق بها فقط (مثل `127.0.0.1,10.0.0.0/8,172.16.0.0/12` في [render.yaml](file:///d:/NourAI/render.yaml)).
- `NOURAI_MAX_BODY_BYTES`: الحد الأقصى لحجم الطلب (`2097152` بايت = `2 MiB`).

---

## 2. مراجعة Docker وRender وTLS
- **[Dockerfile](file:///d:/NourAI/Dockerfile):** يعمل الآن بمستخدم غير جذري `appuser` (`uid 10001`) مع صلاحيات `700` على `/app/data/production` وPython `3.11-slim`.
- **[render.yaml](file:///d:/NourAI/render.yaml):** مضبوط على Python `3.11.9` مع متغيرات البيئة الأمنية.
- **TLS/HTTPS:** ترويسة `Strict-Transport-Security` تُرسل فقط عند `enable_tls=True`، ومفاتيح `.jwt_secret` و`localhost_key.pem` تُحفظ بصلاحيات `0o600`.
- **السجلات والأسرار:** لا تُسجّل كلمات المرور أو التوكنات في مفاتيح `RateLimiter` أو مخرجات الخادم.
