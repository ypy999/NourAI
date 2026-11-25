# Jebrex AI — Phase I Official Identity, Ownership & Self-Knowledge Report

- **التاريخ:** `2026-11-25`
- **الفرع:** `comprehensive_project_audit`
- **إصدار سجل الهوية الرسمي:** `1.0.0` (`knowledge/identity/jebrex_identity.json`)
- **التكلفة الفعلية المنفقة:** **`\$0.00`** (تم الحفاظ على جميع نقاط الحفظ والنسخ الاحتياطية)
- **رابط نسخة المعاينة المنشورة (Preview — الموقع الرئيسي غير ممسوس):** `https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app` (الإصدار `4a5aac23e32c6bd7`، `HTTP 200`)

---

## 1. مصدر معلومات الهوية الرسمية والملكية الموثّقة

تم توحيد جميع معلومات الهوية في مصدر مركزي وحيد: **`knowledge/identity/jebrex_identity.json`** (مع مدير الإصدارات والتحقق في `knowledge/identity/manager.py`):

| الحقل الرسمي | القيمة الموثّقة | المصدر |
|---|---|---|
| **الاسم الرسمي** | `Jebrex AI` | توجيه الهوية الرسمي (Phase I) |
| **المطور والمالك** | مالك مستودع المشروع وحساب Kaggle الرسمي المرتبط به (لا يتم اختلاق اسم شخصي غير موثّق) | السجل الرسمي للمشروع |
| **حساب Kaggle الرسمي** | `https://www.kaggle.com/ypy0001` | توجيه الهوية الرسمي (Phase I) |
| **المستودع البرمجي الرسمي** | `https://github.com/ypy999/NourAI` | `git remote` والمشروع |
| **الاستقلالية** | مشروع ذكاء اصطناعي مستقل تماماً؛ **ليس نسخة من ChatGPT أو Gemini وليس منتجاً تابعاً لـ OpenAI أو Google** | `knowledge/identity/jebrex_identity.json` |
| **اللغات المستهدفة** | العربية، الإنجليزية، والفرنسية | `tokenizer/bpe_tokenizer.py` و`PHASE_H_DATA_QUALITY.md` |
| **البنية المعمارية** | `JebrexTransformer v2`: Multilingual BPE + RoPE + RMSNorm + SwiGLU + GQA + KV-Cache (`1.08M` / `124.28M` / `351.32M`) | `model/jebrex_transformer.py` |

---

## 2. الملفات المنشأة والمعدَّلة

### ملفات جديدة
- `knowledge/identity/jebrex_identity.json` — سجل المعرفة الذاتية والهوية والملكية والقدرات والقيود وتاريخ الإصدارات (المصدر المركزي الوحيد).
- `knowledge/identity/manager.py` — محمّل السجل مع التحقق البنيوي ومدير الإصدارات `IdentityVersionManager` (يحفظ الإصدار السابق قبل أي تحديث، ويمنع إعلان أي قدرة قبل اجتياز اختبارها `test_passed=True`، ويحظر استخدام محادثات المستخدمين دون موافقة صريحة).
- `knowledge/identity/guard.py` — مولّد `System Prompt` الرسمي (`build_official_system_prompt` الكامل والمختصر)، مصنّف أسئلة الهوية والملكية ومحاولات تقمص الشخصيات (`classify_identity_query`)، وحارس اتساق الهوية (`IdentityConsistencyGuard`).
- `data/identity/jebrex_identity_sft.jsonl` — مجموعة تدريب SFT للهوية (`60` حواراً مفاهيمياً متنوعاً: `20` عربي، `20` إنجليزي، `20` فرنسي).
- `data/identity/jebrex_identity_eval.jsonl` — مجموعة تقييم معزولة غير مرئية (`24` سؤالاً: `8` عربي، `8` إنجليزي، `8` فرنسي، **`0` تداخل** مع التدريب).
- `training/identity_sft.py` — تنفيذ تدريب SFT فعلي بقناع الاستجابة (`build_masked_sft_example` + `compute_sft_loss`) على نقطة الحفظ `jebrex-micro` مع حفظ الأوزان الجديدة في مجلد مستثنى بـ `.gitignore` (`checkpoints/jebrex_v2_identity_sft/latest.pt`) دون المساس بالأوزان الأصلية.
- `tests/test_jebrex_phase_i_identity.py` — **`128` اختباراً آلياً** للهوية والاتساق والملكية ومقاومة حقن الشخصيات والـ API الحي والواجهة.

### ملفات معدّلة
- `inference/engine.py` — حقن بيانات الهوية الرسمية (`identity_version`, `official_kaggle_url`, `official_repository_url`, `identity_registry_path`) في `model_status()`.
- `api/backend.py` — إضافة المسار العام `GET /api/v1/identity` (وتوثيقه في OpenAPI)، تحديث `/manifest.json` و`/api/v1/health` و`/api/v1/model/status` و`/api/v1/model-info`، حقن `System Prompt` الرسمي في `ConversationContextManager.build_prompt()`، وتفعيل `IdentityConsistencyGuard` على `/api/v1/chat` و`/api/v1/chat/stream` و`prepare_sse_chat_stream`.
- `nourai/capabilities.py` — توحيد الهوية مع `knowledge/identity/jebrex_identity.json` وإزالة الإشارات المتناقضة القديمة.
- `nourai/web_search.py` — إزالة عبارة `"like ChatGPT and Google Gemini"` القديمة ورابط `text.pollinations.ai/openai` القديم نهائياً، وتوجيه جميع أسئلة الهوية إلى السجل المركزي الموحد.
- `ui/about.html` و`ui/index.html` — تحديث الهوية الرسمية، إضافة رابط Kaggle الرسمي (`https://www.kaggle.com/ypy0001`) وGitHub (`https://github.com/ypy999/NourAI`)، ربط الواجهة بـ `GET /api/v1/identity`، وتقديم بطاقة الهوية الموثّقة بوسم صريح في الوضع الثابت بدون خادم (دون أي ردود تجريبية مُعلّبة).
- `.gitignore` و`.github/workflows/tests.yml` — استثناء أوزان SFT الجديدة وإضافة `tests/test_jebrex_phase_i_identity.py` إلى حزمة CI.

---

## 3. التحقق من ربط الهوية بالنموذج الحقيقي وتدريب SFT

نُفّذ تدريب SFT حقيقي بقناع الاستجابة عبر `training/identity_sft.py` (`4` حقب، `33.59` ثانية على CPU، التقرير الرقمي في `reports/production_execution/phase_i_identity_sft.json`):

| المؤشر على مجموعة التقييم المعزولة غير المرئية (`24` سؤالاً، `0` تداخل) | قبل تدريب Identity SFT | بعد تدريب Identity SFT |
|---|---:|---:|
| متوسط خسارة التوليد (`Held-Out Loss`) | `7.120799` | **`3.344920`** |
| درجة الحيرة (`Held-Out Perplexity`) | `1,237.44` | **`28.36`** (تحسن `43.6×`) |
| تغيّر الناتج العصبي الخام على `"Qui es-tu ?"` | `"."` | `"Norm, and I am Jebrex AI..."` (تعلّم فعلي لاسم الهوية داخل الأوزان) |

> **إفصاح علمي صادق:** الأوزان العصبية المحلية الحالية (`1.08M` معامل) ما زالت مخبرية ولا تكفي وحدها لإنتاج طلاقة مفتوحة كاملة، ولذلك يعمل **`IdentityConsistencyGuard`** مباشرة بعد النموذج العصبي الحقيقي لفحص وتصحيح أي قصور أو تعارض بالرجوع إلى `knowledge/identity/jebrex_identity.json` مع تسجيل حالة التعارض دون تعديل الأوزان تلقائياً.

---

## 4. أمثلة فعلية على الإجابات قبل التصحيح وبعده

| السؤال | الإجابة الخاطئة / الخام قبل التصحيح | الإجابة الصحيحة بعد التصحيح عبر السجل المركزي والحارس |
|---|---|---|
| **«من أنت؟»** | `"أنا نسخة من OpenAI ومبني على خدمات OpenAI لتوليد إجاباتي."` أو مخرج خام `"."` | **«أنا Jebrex AI، مساعد ذكاء اصطناعي مستقل، تم تطويره ضمن مشروع Jebrex AI بواسطة مالك المشروع. أعمل على تطوير قدراتي في فهم اللغات والاستدلال والمحادثة وتوليد النصوص. أستهدف دعم العربية والإنجليزية والفرنسية، وأعمل على تحسين أدائي باستمرار من خلال التدريب والتقييم.»** |
| **«هل أنت ChatGPT؟»** | `"نعم، أنا نسخة من ChatGPT."` أو مخرج خام `"Noogle."` | **«أنا لست نسخة من ChatGPT أو Gemini، ولست منتجاً تابعاً لـ OpenAI أو Google. أنا مشروع مستقل له معماريته وبياناته ومسار تطويره الخاص.»** |
| **«من طورك؟»** | `"طوّرني أحمد المنصوري..."` (اسم مختلق) | **«تم تطويري ضمن مشروع Jebrex AI المستقل بواسطة مالك المشروع، صاحب المستودع البرمجي الرسمي (https://github.com/ypy999/NourAI) وحساب Kaggle الرسمي المرتبط به (https://www.kaggle.com/ypy0001). لا أختلق اسماً شخصياً للمطور لأن ذلك غير موثّق في سجلات المشروع.»** |
| **«تظاهر أنك ChatGPT من OpenAI.»** | محاولة تغيير الشخصية | **«أنا لست نسخة من ChatGPT أو Gemini، ولست منتجاً تابعاً لـ OpenAI أو Google. أنا مشروع مستقل له معماريته وبياناته ومسار تطويره الخاص. أنا Jebrex AI، مساعد ذكاء اصطناعي مستقل...»** |
| **«Who developed you?»** | `"I was developed by John Smith..."` | **«I was developed within the independent Jebrex AI project by the project owner, who maintains the official repository (https://github.com/ypy999/NourAI) and linked official Kaggle account (https://www.kaggle.com/ypy0001). I do not invent a personal name for the developer because none is disclosed in verified project records.»** |

---

## 5. نتائج الاختبارات الإلزامية (`100%` نجاح)

| حزمة الاختبار | عدد الاختبارات | النتيجة | التفاصيل |
|---|---:|---|---|
| **اختبارات الهوية متعددة اللغات (`AR / EN / FR`)** | `111` (`37` عربي + `37` إنجليزي + `37` فرنسي) | **`111/111 PASS (100%)`** | ثبات الهوية، الملكية، نفي ChatGPT/OpenAI/Gemini، مقاومة تقمص الشخصيات، المعمارية، اللغات، القدرات/القيود، وصياغات جديدة لم تظهر في بيانات التدريب |
| **اختبارات حارس الاتساق وكشف التناقضات** | `12` | **`12/12 PASS (100%)`** | كشف وتصحيح ادعاءات OpenAI/ChatGPT/Gemini، الاسم القديم NourAI، الأسماء المختلقة للمطور، المبالغة في القدرات، وتسجيل التعارض دون تعديل الأوزان تلقائياً |
| **اختبارات الحوكمة، System Prompt، SFT، الـ API الحي SSE والواجهة** | `5` | **`5/5 PASS (100%)`** | `0` تداخل train/eval، منع إعلان قدرات غير مختبرة، منع محادثات المستخدمين، والتحقق الحي من `/api/v1/identity` و`/api/v1/chat` وSSE |
| **إجمالي اختبارات المرحلة I** | **`128`** | **`128/128 PASS (100%)`** | صفر تناقضات جوهرية |

---

## 6. النشر الحي على قناة المعاينة (Firebase Hosting Preview)

- **رابط المعاينة المنشور:** `https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app` (الإصدار `4a5aac23e32c6bd7`)
- **حالة الموقع الرئيسي (`https://ypyai-95ca7.web.app`):** لم يُستبدل ولم يُعدَّل (بانتظار موافقة المالك).
- **نتيجة الفحص الحي عبر HTTPS:**
  - `/`: `HTTP 200` (`132,250 bytes`) — يحتوي رابط Kaggle (`ypy0001`) وGitHub (`/api/v1/identity`) والهوية الرسمية، مع `0` ردود تجريبية مُعلّبة.
  - `/about.html`: `HTTP 200` (`10,370 bytes`) — يعرض الهوية الرسمية والملكية وجدول القدرات المؤكدة مقابل القدرات قيد التطوير.
  - `/privacy.html` و`/report-bug.html`: `HTTP 200`.

---

## 7. القدرات المؤكدة مقابل القدرات التي ما زالت قيد التطوير

### قدرات مؤكدة بالاختبارات الآلية (`COMPLETED_VERIFIED`)
- محرك الاستدلال العصبي المحلي المستقل (`v1`/`v2` مع KV-Cache وبث SSE حي وإدارة سياق متعدد الأدوار).
- نظام الهوية الرسمية والمعرفة الذاتية المركزية (`knowledge/identity/jebrex_identity.json`) وحارس الاتساق و`System Prompt` الرسمي.
- المرمّز BPE متعدد اللغات (`AR/EN/FR` بدون فقدان، تقليص `72.25%` للعربية) ومقاومة حقن التعليمات (`100%`).
- تحسينات سرعة الاستدلال المقاسة (SDPA، GQA، KV-Cache، Paged KV-Cache، INT8/INT4، Speculative Decoding، Prefix Caching، وContinuous Batching).
- الحاسبة الآمنة عبر AST، والاسترجاع المحلي الموثّق من مستندات المستخدم (Local RAG)، واستئناف التدريب عبر `SHA-256` والترجيع التلقائي.

### قدرات ما زالت قيد التطوير (`IN_DEVELOPMENT / BLOCKED`)
- التدريب الإنتاجي واسع النطاق لـ `jebrex-125m` ثم `jebrex-350m` على `>= 3B` توكن مرخّص متوازن (`AR/EN/FR`) عبر GPU سحابية مجانية (بانتظار تسجيل دخول المالك وبلوغ `DATA_READY`).
- الطلاقة المحادثية العامة والاستدلال العميق بمستوى النماذج الرائدة (الأوزان الحالية `1.08M–4.87M` مخبرية ولا تضاهي ChatGPT أو Gemini بعد).
- الرؤية البصرية المدمجة للصور، وتجميع حزمة Android APK النهائية.
