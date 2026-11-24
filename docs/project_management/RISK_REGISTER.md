# Jebrex AI — Risk Register (سجل المخاطر — محدّث بعد المعالجة)

| المعرف | الفئة | وصف الخطر | الأولوية | الحالة النهائية | الإجراء المنفّذ / المتبقي |
|---|---|---|---|---|---|
| `RSK-01` | أمني | تنفيذ كود عشوائي (`RCE`) عبر `/api/v1/code/run` | `CRITICAL` → مغلق | `COMPLETED_VERIFIED` | أُلغي `subprocess` واستُبدل بمقيّم AST حسابي خالص (`test_jebrex_security.py`) |
| `RSK-02` | أمني | تصعيد صلاحيات فوري إلى `admin` عبر `/api/v1/users/register` | `CRITICAL` → مغلق | `COMPLETED_VERIFIED` | إرجاع `403` لأي دور مميز وقفل التسجيل العام على `user` |
| `RSK-03` | أمني | تخزين مفتاح Gemini في Cookie نصي وغياب `HttpOnly` | `HIGH` → مغلق | `COMPLETED_VERIFIED` | إزالة Cookie المفتاح وإضافة `HttpOnly` ومحاذاة `Max-Age` مع TTL |
| `RSK-04` | استقلالية | تمرير ردود مزود خارجي وكأنها من النموذج الأساسي | `HIGH` → مغلق | `COMPLETED_VERIFIED` | الوضع العصبي المحلي افتراضي 100% وتعطيل `allow_external_llm` افتراضياً |
| `RSK-05` | بيانات | تدريب النموذج على تشويش OCR (`27.8%`) أو تلوث اختبار | `HIGH` → محجوز بالبوابة | `COMPLETED_VERIFIED` | `data/jebrex_data_gate.py` يمنع التدريب ما لم تجتز البيانات البوابة |
| `RSK-06` | موارد | نفاد مساحة `C:` (`6.95 GiB`) أو الذاكرة (`2.51 GiB`) | `HIGH` → مضبوط | `COMPLETED_VERIFIED` | فحص `preflight_resource_check`؛ إجمالي الإضافات `< 25 MiB` |
| `RSK-07` | معماري | ضعف كفاءة `ByteTokenizer` مع العربية وضيق `context=256` | `MEDIUM` | `BLOCKED` | يتطلب إعادة تدريب كاملة بـ `Subword BPE` على GPU (`LEVEL 1`) |
| `RSK-08` | تشغيلي | إغراق الذاكرة عبر `Content-Length` وغياب `client_ip` الحقيقي | `HIGH` → مغلق | `COMPLETED_VERIFIED` | فرض سقف `1 MiB` (`413`) وتمرير `self.client_address[0]` ومعالجة `JSONDecodeError` (`400`) |
| `RSK-09` | أمني | السماح بـ `'unsafe-inline'` في CSP، وحفظ `.jwt_secret` نصياً محلياً، وحقن الأوامر المموّه | `MEDIUM` | `IN_PROGRESS` | موثقة في `BLOCKERS_AND_RISKS.md` و`security_audit_report.json` لمعالجتها قبل النشر العام |
