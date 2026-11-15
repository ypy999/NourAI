# تقرير الفحص الجنائي والمراجعة الهندسية المستقلة (`2026-11-14`)

- **تاريخ الفحص:** `2026-11-14`
- **المسار:** `D:\NourAI`
- **منهجية الفحص:** تشغيل فعلي للاختبارات (`261/261`)، وتوليد مباشر من نقطة الحفظ `checkpoints/v1_2/nourai_v1_2_model.pt` عبر `InferenceEngine.generate`، وفحص البصمات الرقمية `SHA-256` والتقسيمات ومسارات الكود الفعلي في `api/backend.py` و`nourai/capabilities.py`.

---

## 1. جدول الحكم الهندسي الصريح

| المحور | التقييم قبل الإصلاح | التقييم بعد الإصلاح (`2026-11-14`) |
|---|---|---|
| **جودة الكود والاختبارات** | 🟢 جيدة (`261/261` اختبار ناجح) | 🟢 ممتازة وموثقة مع `--basetemp` |
| **هندسة البيانات ومنع التسرب** | 🟢 سليمة (`0` تقاطع نصوص أو مستندات أو SFT) | 🟢 سليمة ومُتحقَّق منها (`ar 33.66% / en 33.25% / fr 33.10%`) |
| **صدق الربط بين الـAPI والنموذج** | 🔴 كان `use_capabilities=True` يحجب النموذج العصبي ويعرض مقاييسه مع ردود قوالب | 🟢 تم الإصلاح: `use_capabilities=False` افتراضياً (`response_source="neural_model"`) وفصل مقاييس القوالب تماماً |
| **إدارة الأسرار والأمان و`.gitignore`** | 🟠 مفتاح ثابت في الكود، `/auth/guest` بلا حد معدل، و`data/production/` غير مستثنى | 🟢 تم الإصلاح: المفتاح من `NOURAI_SECRET_KEY` أو `secrets.token_hex(32)`، تحديد معدل `/auth/guest`، وإضافة `data/production/` إلى `.gitignore` |
| **النموذج اللغوي العصبي (`843K` معامل)** | 🔴 تدريب قصير جداً (`60+30` خطوة) ينتج نصوصاً عشوائية | 🟡 نموذج مصغّر (`Nano 843K`) بعد تدريب موسّع مقنّع الخسارة؛ يجيب على أنماط التدريب المحدودة لكنه ليس نموذجاً عاماً بحجم المليارات |

---

## 2. الإصلاحات المنفذة فوراً استجابةً للفحص

1. **إصلاح صدق المنتج في [backend.py](file:///d:/NourAI/api/backend.py) و[index.html](file:///d:/NourAI/ui/index.html) و[run_nourai.py](file:///d:/NourAI/run_nourai.py):**
   - جعل `use_capabilities: bool = False` هو الوضع الافتراضي في `NourAIBackend`.
   - في الوضع الافتراضي، يستدعي `/api/v1/chat` و`/api/v1/chat/stream` محرك `InferenceEngine` مباشرةً ويُرجع `response_source: "neural_model"` مع مقاييس التوليد الحقيقية.
   - عند طلب المساعد القاعدي صراحةً (`use_fallback_assistant: true`)، لا يتم تشغيل الشبكة العصبية ولا تُسرق مقاييسها، بل يُرجع `response_source: "rule_based_fallback_assistant"` مع `generated_tokens: 0`.
2. **إصلاح الثغرات الأمنية وملف [.gitignore](file:///d:/NourAI/.gitignore):**
   - إزالة السلسلة الثابتة `secret_key="nourai-phase18-local-secret-key"` واستبدالها بقراءة `os.environ.get("NOURAI_SECRET_KEY")` أو توليد مفتاح عشوائي آمن `secrets.token_hex(32)` لكل نسخة.
   - إخضاع المسار `/api/v1/auth/guest` لمحدد المعدل `RateLimiter` (`guest:{client_ip}`) قبل إنشاء أي حساب ضيف.
   - إضافة `data/production/` و`data/backups/` إلى `.gitignore`.
3. **تصحيح التوثيق في [README.md](file:///d:/NourAI/README.md) و[ARCHITECTURE.md](file:///d:/NourAI/docs/ARCHITECTURE.md) و[requirements.txt](file:///d:/NourAI/requirements.txt):**
   - تصحيح اسم ملف المعمارية إلى `model/transformer.py`.
   - تصحيح معرّفات الرموز الخاصة إلى `256:<PAD>, 257:<BOS>, 258:<EOS>, 259:<UNK>` والبايتات `0..255`.
   - حذف الحزمتين غير المستخدمتين `fastapi` و`uvicorn` من `requirements.txt`.
