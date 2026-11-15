# تقرير التدقيق الجنائي المستقل الشامل لمشروع NourAI (Stage A — Read-Only Forensic Audit)

- **تاريخ التدقيق:** `2026-11-13`
- **مسار المستودع:** `D:\NourAI`
- **وضع التدقيق:** قراءة فقط بالكامل (`STRICTLY READ-ONLY` — لم يُعدَّل أو يُحذف أي ملف أصلي، ولم يُشغَّل أي تدريب أو `CUDA`)
- **نتيجة المرحلة:** `FINAL_AUDIT_COMPLETE`
- **الحكم الهندسي العام:** **البنية التحتية والمعمارية وخط أنابيب البيانات (Phases 0–12) حقيقية وموثقة برمجياً (`253/253` اختباراً ناجحاً)، لكن مراحل التدريب الفعلي والتوليد اللغوي وتكامل الواجهة والإنتاج (Phases 13–24) تعاني من فجوات جوهرية تمنع اعتبار النموذج الحالي جاهزاً للاستخدام اللغوي الحقيقي دون تنفيذ مراحل المعالجة (Stages B–L).**

---

## المحور الأول: التدقيق الشامل لبنية المشروع والملفات والبيئة (Axis 1)

### 1.1 فحص الهيكل العام للمجلدات والملفات
أُجري مسح شامل لكافة ملفات المستودع (باستثناء `.venv` و`.git` و`__pycache__`)، وبلغ إجمالي الملفات **`2,034` ملفاً** موزعة على **`18` مجلداً رئيسياً**:

| البند | الحالة على القرص | التفاصيل والأدلة |
| :--- | :---: | :--- |
| المجلدات الرئيسية الموجودة (`18`) | `VERIFIED` | `api/`, `checkpoints/`, `configs/`, `corpus/`, `data/`, `datasets/`, `deployment/`, `docs/`, `evaluation/`, `inference/`, `model/`, `nourai/`, `releases/`, `reports/`, `tests/`, `tokenizer/`, `training/`, `ui/` |
| عناصر مفقودة مقارنة بالهيكل المرجعي (`5`) | `MISSING` | `security/` (مدمج حالياً في [security_performance.py](file:///d:/NourAI/api/security_performance.py))، `scripts/`، `requirements/` (يوجد ملف [requirements.txt](file:///d:/NourAI/requirements.txt) في الجذر)، `LICENSE`، `pyproject.toml` |
| ملفات فارغة غير `__init__.py` (`21` ملفاً) | `WARNING` | أبرزها [tatoeba_cc0_multilingual.jsonl](file:///d:/NourAI/data/processed/phase8/imported/tatoeba_cc0_multilingual.jsonl) (`0` بايت)، وملف [write_test.tmp](file:///d:/NourAI/reports/write_test.tmp) (`0` بايت)، و`3` ملفات `.sqlite3-wal` في `.backup/`، و`16` ملفاً مؤقتاً قديماً تحت `reports/phase10/pytest_temp/` |
| صلاحية ملفات الإعدادات `configs/*.json` (`13` ملفاً) | `VERIFIED` | جميع ملفات JSON الـ `13` في `configs/` سليمة وتقرأ بترميز `UTF-8` دون أي خطأ نحوي |

### 1.2 البيئة البرمجية والاعتماديات ونتيجة الاختبارات
- **نظام التشغيل:** `Windows-10-10.0.19045-SP0`
- **إصدار Python:** `3.15.2 (AMD64)`
- **المكتبات المثبتة:** `torch 2.15.0`، `pytest 9.2.0`، `numpy 2.5.4`، `sqlite3 3.53.4`
- **تشغيل الاختبارات المعزول:**
  - الأمر: `pytest -q -p no:cacheprovider --basetemp=<isolated_scratch>`
  - النتيجة: **`253 passed in 22.90s`** (`207` اختبارات للمراحل `0–10.12` + `46` اختباراً للمراحل `11–24`).

### 1.3 مطابقة تقارير المراحل (`Phase 0` إلى `Phase 24`) مع الواقع الفعلي

| المرحلة | التقرير المعلن | الحكم الجنائي المستقل | الدليل الفعلي من الكود والملفات |
| :--- | :---: | :---: | :--- |
| **Phases 0–9** | `PASS` | **حقيقي ومكتمل (`VERIFIED`)** | البنية الأساسية، التوكنايزر، نموذج GPT، محرك التدريب، وقاعدة البيانات الأولية تعمل فعلياً ومختبرة. |
| **Phases 10.3–10.12** | `CERTIFIED_READY` | **حقيقي مع خلل توازن لغوي في التوكنات (`VERIFIED_WITH_CAVEAT`)** | استيراد حقيقي لـ `Tatoeba` و`QNL` (`2%`) و`WTO` (`2%`)، صفر تسرب بين التقسيمات (`0` leakage)، لكن العربية تشكل **`76.94%`** من توكنات التدريب رغم أنها `50` سجلاً فقط (كتب كاملة مقابل جمل قصيرة للغات الأخرى). |
| **Phase 11** | `CERTIFIED` | **حقيقي ومكتمل (`VERIFIED`)** | معمارية `GPT` (`843,008` معاملات) مطابقة تماماً للمواصفات في [gpt.py](file:///d:/NourAI/model/gpt.py). |
| **Phase 12** | `CERTIFIED` | **حقيقي ومكتمل (`VERIFIED`)** | [trainer.py](file:///d:/NourAI/training/trainer.py) و[engine_audit.py](file:///d:/NourAI/training/engine_audit.py) يختبران تجميع التدرجات، القص، الجدولة، والاستئناف الحقيقي. |
| **Phase 13** | `CERTIFIED` | **تدريب حقيقي لكنه مصغر جداً (`PARTIAL`)** | في [initial_training.py](file:///d:/NourAI/training/initial_training.py)، تم أخذ `540` سجلاً فقط (`198,169` توكن = **`0.286%`** من بيانات `phase10_3`) والتدريب لـ **`40` خطوة فقط**. |
| **Phase 14** | `CERTIFIED` | **تدريب حقيقي (`30` خطوة) مع ضياع حالة المُحسِّن (`PARTIAL`)** | في [optimization_scaling.py](file:///d:/NourAI/training/optimization_scaling.py#L255)، تم تحميل الأوزان فقط عبر `GPT.load_checkpoint` وإنشاء `Trainer` جديد بصفر خطوات بدلاً من `Trainer.resume(last.pt)`. |
| **Phase 15** | `CERTIFIED` | **قياسات حقيقية لكن التوليد غير مترابط (`PARTIAL`)** | حسابات `Loss` و`Perplexity` حقيقية، لكن النصوص المولدة في [phase15_multilingual_evaluation_report.json](file:///d:/NourAI/reports/phase15/phase15_multilingual_evaluation_report.json) مجرد مقاطع حروف عشوائية (`"الالالال"`، `"e nerre seeh o e"`). |
| **Phase 16** | `CERTIFIED` | **بيانات قوالب اصطناعية مكررة وتسرب تقييم (`SYNTHETIC`)** | في [finetuning_alignment.py](file:///d:/NourAI/training/finetuning_alignment.py#L31-L158)، البيانات مبنية من `18` قالباً مكرراً `16` مرة (`288` سجلاً)، واختبار التقييم يحتوي على نفس الإجابات حرفياً. |
| **Phase 17** | `CERTIFIED` | **حقيقي مع نقص `stop_sequences` (`VERIFIED_PARTIAL`)** | [engine.py](file:///d:/NourAI/inference/engine.py) يدعم `top_k`/`top_p`/`temperature` والبث بترميز `UTF-8`، لكنه يفتقر إلى `stop_sequences` وإيقاف البث الخارجي. |
| **Phase 18** | `CERTIFIED` | **حقيقي في الذاكرة فقط (`PARTIAL`)** | [backend.py](file:///d:/NourAI/api/backend.py) يخزن المستخدمين والجلسات في قواميس `dict` داخل الذاكرة العشوائية (`RAM`) وتُفقد عند إعادة التشغيل. |
| **Phase 19** | `CERTIFIED` | **خلل تكامل حرج مع الواجهة (`BROKEN_AUTH_FLOW`)** | [index.html](file:///d:/NourAI/ui/index.html#L222-L235) يرسل طلب `/api/v1/chat/stream` بدون ترويسة `Authorization: Bearer`، فيرفضه الخادم بـ `401 Unauthorized`. |
| **Phase 20** | `CERTIFIED` | **حقيقي (`VERIFIED_IN_PROCESS`)** | تشفير `PBKDF2-HMAC-SHA256`، توكنات `HMAC`، تحديد المعدل، وفلاتر `XSS`/`SQLi` تعمل فعلياً في [security_performance.py](file:///d:/NourAI/api/security_performance.py). |
| **Phase 21** | `CERTIFIED` | **HTTP عادي وليس HTTPS فعلياً (`PARTIAL`)** | [production.py](file:///d:/NourAI/deployment/production.py#L223-L250) يشغل `ThreadingHTTPServer` بدون `ssl_context.wrap_socket`، والجلسات في `NourAIBackend` غير مربوطة بجدول `prod_sessions`. |
| **Phase 22** | `CERTIFIED` | **حزم إصدار حقيقية (`VERIFIED_ARTIFACTS`)** | ملفات `releases/v1.0.0/` موجودة ومطابقة لبصمة `SHA-256` لنموذج المرحلة 16. |
| **Phase 23** | `CERTIFIED` | **صيانة وحوادث مُحاكاة برمجياً (`SIMULATED`)** | في [release_and_lifecycle.py](file:///d:/NourAI/deployment/release_and_lifecycle.py#L204-L238)، الحادثة `TRANSIENT_TIMEOUT_WARN` والتقييمات الـ `3` مكتوبة بشكل ثابت (`hardcoded`). |
| **Phase 24** | `CERTIFIED` | **تدريب مستمر (`16` خطوة) على نفس القوالب الـ 18 (`SYNTHETIC`)** | [release_and_lifecycle.py](file:///d:/NourAI/deployment/release_and_lifecycle.py#L276-L360) يكرر نفس الـ `18` قالباً من المرحلة 16 لـ `16` خطوة فقط لإنتاج `v1.1.0`. |

---

## المحور الثاني: تدقيق البيانات Corpus والتقسيمات (Axis 2)

### 2.1 المصادر الخام والترخيص وبصمات `SHA-256`
من أصل `14` مصدراً مسجلاً في [dataset_sources.json](file:///d:/NourAI/configs/dataset_sources.json)، تم فعلياً تنزيل واستيراد **`3` مصادر حقيقية**:
1. **`tatoeba_cc0`**:
   - المسار: [tatoeba_cc0.tar.bz2](file:///d:/NourAI/data/raw/external/tatoeba_cc0.tar.bz2) (`7,963,537` بايت)
   - `SHA-256`: `7e4251d06edd949db9e47cc360b472873da6c054af5dc4b9d2c9e06d024cd710`
   - السجلات المستوردة: `55,719` جملة متعدّدة اللغات (`CC0-1.0`).
2. **`qnl_arabic_ocr_v2` (مكتبة قطر الوطنية)**:
   - المسار: [qnl_arabic_ocr_v2.zip](file:///d:/NourAI/data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip) (`1,065,878,922` بايت) + [qnl_metadata.csv](file:///d:/NourAI/data/raw/external/phase10_2/qnl_metadata.csv) (`1,887,609` بايت)
   - `SHA-256 (ZIP)`: `352d7d2e1e3c64255a52b09560da97bb0360e43a327bd70298c7fa52b2437cb9`
   - ملاحظة جوهرية: تم أخذ عينة **`2%` فقط** (`58` كتاباً، بقيت منها `52` وثيقة بعد التنظيف)، وكل كتاب مخزن كسجل واحد ضخم يصل إلى مئات الآلاف من الأحرف!
3. **`wto_public_corpus_en_fr` (منظمة التجارة العالمية)**:
   - المسار: [wto_en_fr_corpus_2019_EN-FR.zip](file:///d:/NourAI/data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip) (`1,501,806,179` بايت)
   - `SHA-256`: `09f958e84a626cc294eaa0c3b87b2daa7abf1f786fc91d2978756264750545f5`
   - السجلات المستوردة: `58,466` فقرة/جملة من عينة `2%`.

### 2.2 التفاوت الحاد بين عدد السجلات وعدد التوكنات في `phase10_3` و`phase13`

| المجموعة / القسم | اللغة | عدد السجلات | عدد الأحرف (`Chars`) | عدد التوكنات (`Byte Tokens`) | نسبة التوكنات الفعلية |
| :--- | :---: | ---: | ---: | ---: | ---: |
| **`phase10_3/splits/train.jsonl`** | العربية (`ar`) | `50` | `30,205,797` | `53,305,428` | **`76.94%`** |
| **`phase10_3/splits/train.jsonl`** | الإنجليزية (`en`) | `58,214` | `3,746,458` | `6,363,901` | **`9.19%`** |
| **`phase10_3/splits/train.jsonl`** | الفرنسية (`fr`) | `35,036` | `2,367,652` | `5,075,783` | **`7.33%`** |
| **`phase10_3/splits/train.jsonl`** | أخرى (`other` - كتب QNL ثنائية) | `4` | `2,659,788` | `4,535,630` | **`6.55%`** |
| **`phase13/splits/train.jsonl`** *(المستخدمة فعلياً)* | العربية (`ar`) | `180` | `92,160` | `166,006` | **`83.77%`** |
| **`phase13/splits/train.jsonl`** *(المستخدمة فعلياً)* | الإنجليزية (`en`) | `180` | `13,537` | `22,587` | **`11.40%`** |
| **`phase13/splits/train.jsonl`** *(المستخدمة فعلياً)* | الفرنسية (`fr`) | `180` | `9,558` | `9,576` | **`4.83%`** |

> [!IMPORTANT]
> **النتيجة الجنائية للمحور الثاني:**
> 1. على الرغم من أن `phase13` ساوى عدد السجلات (`180` سجلاً لكل لغة)، إلا أن السجل العربي الواحد يبلغ `512` حرفاً (`~922` بايت/توكن لأن الحرف العربي يأخذ `2` بايت في `UTF-8`)، بينما السجل الإنجليزي أو الفرنسي من `Tatoeba` مجرد جملة قصيرة (`~53–75` حرفاً). النتيجة هي أن **`83.77%` من توكنات التدريب الفعلي كانت عربية**، و**`4.83%` فقط فرنسية**!
> 2. لم يستخدم التدريب في المرحلتين 13 و14 سوى **`198,169` توكن** من أصل **`69,280,742` توكن** متوفرة في `phase10_3/splits/train.jsonl` (أي **`0.286%` فقط** من البيانات المعالجة).

---

## المحور الثالث: تدقيق المُرمِّز النصي `ByteTokenizer` (Axis 3)

- **الملف:** [byte_tokenizer.py](file:///d:/NourAI/tokenizer/byte_tokenizer.py)
- **حجم المفردات (`vocab_size`):** `260` (`0:<PAD>`, `1:<BOS>`, `2:<EOS>`, `3:<UNK>`, `4..259` لبايتات `0x00..0xFF`).
- **اختبار الترميز وفك الترميز (`Round-Trip`):**
  - تم التحقق جنائياً من تطابق `decode(encode(text)) == text` بنسبة **`100%`** على العربية (مع التشكيل والهمزات)، الإنجليزية، الفرنسية (مع الحروف المعجمة `é, è, ê, ç, œ`)، النصوص المختلطة، الأرقام، علامات الترقيم، والرموز الخاصة.
  - يتعامل بأمان مع البايتات المقطوعة عبر `bytes.decode("utf-8", errors="replace")`.
- **نقطة الاختناق المعمارية:**
  - نظراً لأن الحرف العربي يستهلك `2` بايت في `UTF-8`، فإن معدل التوكنات لكل حرف عربي هو **`1.80` توكن/حرف** مقابل **`1.00` توكن/حرف** للإنجليزية. هذا يعني أن نافذة السياق `context_length = 128` لا تتسع إلا لحوالي **`70` حرفاً عربياً فقط** (أقل من سطرين).

---

## المحور الرابع: تدقيق معمارية النموذج `GPT` (Axis 4)

- **الملف:** [gpt.py](file:///d:/NourAI/model/gpt.py)
- **التحقق من المعاملات والأبعاد:**
  - النموذج الافتراضي (`Default` المعتمد في المراحل 11–24): `n_layers=4`, `n_heads=4`, `d_model=128`, `d_ff=512`, `context_length=128`, `vocab_size=260` ← إجمالي المعاملات **`843,008`** (`3.22 MB` بـ `fp32`).
  - ملف التوسع الصغير (`Small` في المرحلة 14): `6` طبقات، `d_model=256` ← **`4,871,168` معاملاً** (`18.58 MB`).
  - ملف التوسع الأساسي (`Base` في المرحلة 14): `8` طبقات، `d_model=512` ← **`25,615,360` معاملاً** (`97.71 MB`).
- **سلامة التنفيذ:** القناع السببي (`Causal Mask`)، التطبيع القبلي (`Pre-LayerNorm`)، تهيئة الأوزان (`Normal(0, 0.02)` مع تحجيم الطبقات المتبقية بـ $1/\sqrt{2 N_{\text{layers}}}$)، وحفظ واستعادة نقاط الاستعادة تعمل جميعها بدقة كاملة.

---

## المحور الخامس: التدقيق الجنائي لمحرك التدريب وسجل التدريب (Axis 5)

### 5.1 تتبع خطوات التدريب الفعلية عبر المراحل (13 ← 14 ← 16 ← 24)

| المرحلة | ملف نقطة الاستعادة | عدد الخطوات | التوكنات المعالجة | `Val Loss` (البداية ← النهاية) | هل استُعيدت حالة `AdamW` و`Scheduler`؟ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 13** | [phase13_model.pt](file:///d:/NourAI/checkpoints/phase13/phase13_model.pt) | `40` | `81,920` | `5.5449` ← `3.0125` | بداية جديدة (`Step 0 -> 40`) |
| **Phase 14** | [phase14_optimized_model.pt](file:///d:/NourAI/checkpoints/phase14/phase14_optimized_model.pt) | `30` | `61,440` | `3.0125` ← `1.9675` | **لا (`RESET` إلى `Step 0`)** |
| **Phase 16** | [phase16_sft_aligned_model.pt](file:///d:/NourAI/checkpoints/phase16/phase16_sft_aligned_model.pt) | `32` | `40,960` | `3.8100` ← `2.4046` | **لا (`RESET` إلى `Step 0`)** |
| **Phase 24** | [nourai_v1_1_model.pt](file:///d:/NourAI/checkpoints/phase24/nourai_v1_1_model.pt) | `16` | `20,480` | `2.4007` ← `2.0847` | **لا (`RESET` إلى `Step 0`)** |

### 5.2 المخالفات الجنائية المكتشفة في مسار التدريب
1. **إعادة تصفير حالة المُحسِّن (`AdamW`) والجدولة (`Scheduler`) بين المراحل:**
   - يحفظ `Trainer` الحالة الكاملة (بما فيها `optimizer_state` و`scheduler_state` و`global_step`) داخل `last.pt` (`10.19 MB`)، بينما يحفظ `GPT.save_checkpoint()` الأوزان فقط (`3.39 MB`).
   - في [optimization_scaling.py](file:///d:/NourAI/training/optimization_scaling.py#L255) و[finetuning_alignment.py](file:///d:/NourAI/training/finetuning_alignment.py#L250) و[release_and_lifecycle.py](file:///d:/NourAI/deployment/release_and_lifecycle.py#L346)، تم استدعاء `GPT.load_checkpoint()` وإنشاء كائن `Trainer` جديد كلياً في كل مرحلة، مما أدى إلى تصفير عزوم `AdamW` (`m_t, v_t`) وإعادة `global_step` إلى الصفر في بداية كل مرحلة.
2. **الحجم الضئيل جداً للتدريب الكلي:**
   - مجموع خطوات تحديث الأوزان عبر تاريخ المشروع بالكامل هو **`118` خطوة فقط** (`204,800` توكن بايت).
3. **التكرار الاصطناعي وتسرب التقييم في المرحلتين 16 و24:**
   - في [finetuning_alignment.py](file:///d:/NourAI/training/finetuning_alignment.py#L31-L158)، تتكون بيانات `phase16` من `18` قالباً ثابتاً مكرراً `16` مرة في التدريب (`288` سجلاً) و`4` مرات في التحقق والاختبار (`72` سجلاً) مع نفس نص الإجابة (`response`) تماماً!

---

## المحور السادس والسابع: تدقيق القدرات اللغوية (العربية، الإنجليزية، الفرنسية) وقدرات الذكاء الاصطناعي (Axes 6 & 7)

### 6.1 الفرق بين مقاييس `Teacher-Forced Perplexity` وجودة التوليد الفعلي (`Autoregressive Generation`)
عند فحص مخرجات التوليد الحتمي (`Greedy Decoding, top_k=1`) لجميع نقاط الاستعادة الأربع على نفس المطالبات القياسية، ظهرت الحقيقة الجنائية التالية:

| نقطة الاستعادة | مطالبة عربية (`"اللغة العربية هي"`) | مطالبة إنجليزية (`"Artificial intelligence is"`) | مطالبة فرنسية (`"L'intelligence artificielle est"`) |
| :--- | :--- | :--- | :--- |
| **`phase13_initial`** | `"الالالالالالالال"` | `"                                "` | `"                                "` |
| **`phase14_optimized`** | `"الالالالالالالال"` | `"e e e e e e e e e e e e e e e e "` | `"e e e e e e e e e e e e e e e e "` |
| **`phase16_sft_aligned` (`v1.0.0`)** | `"الالالالالالالال"` | `"<\nالالالالالالالا"` | `"<\nالالالالالالالا"` |
| **`phase24_continual` (`v1.1.0`)** | `"الالالالالالالال"` | `"<||||||||||||||||||||||||||||||a"` | `"<||||||||||||||||||||||ase s le "` |

> [!WARNING]
> **تفسير الفجوة العلمية:**
> انخفاض `Perplexity` من `255.93` إلى `7.14` في المرحلة 14 حقيقي رياضياً (لأن النموذج تعلم التوزيع الإحصائي للبايتات الأكثر تكراراً مثل بايتات `"ال"` في العربية والمسافات وحرف `"e"` في الإنجليزية والفرنسية)، لكن **`118` خطوة تدريب على نموذج بمستوى البايت (`Byte-level`) لا تكفي إطلاقاً لتكوين كلمات صحيحة أو جمل مفهومة**. كما أن التدريب على قوالب المرحلة 16 و24 جعل النموذج يكرر رمز `<` أو `|` من وسم `<|assistant|>` متبوعاً بـ `"الالالال"` حتى عند سؤاله بالإنجليزية أو الفرنسية!

### 6.2 تقييم مصفوفة قدرات الذكاء الاصطناعي (Axis 7)
- **ما يعمل فعلياً كطبقة برمجية (`System/Wrapper Level`):**
  - إدارة الجلسات متعددة الأدوار (`SessionManager`)، حقن السياق (`context_messages`)، وفلتر السلامة المسبق (`evaluate_prompt_safety`) الذي يرفض الطلبات الضارة برسائل سليمة لغوياً بالعربية والإنجليزية والفرنسية.
- **ما لا يعمل فعلياً على مستوى أوزان النموذج (`Model Weights Level`):**
  - الفهم القرائي، الإجابة عن الأسئلة، التلخيص، الترجمة بين اللغات الثلاث، كتابة الأكواد، حل المسائل الرياضية، والاستدلال المنطقي كلها **غير متحققة حالياً** بسبب عدم قدرة النموذج المدرب لـ `118` خطوة على توليد كلمات مترابطة.
- **التقييم البشري (`Human Evaluation`):** لا يوجد أي بروتوكول أو سجل تقييم بشري (`Rubrics` / `Blind Rating`) في المستودع.

---

## المحور الثامن والتاسع: تدقيق محرك الاستدلال، الواجهة البرمجية `API`، وواجهة المستخدم `UI` (Axes 8 & 9)

### 8.1 محرك الاستدلال ([inference/engine.py](file:///d:/NourAI/inference/engine.py))
- **المزايا المتحققة:** تحميل نقاط الاستعادة مع `Warm Cache`، دعم `greedy` و`top_k` و`top_p` و`temperature` و`repetition_penalty`، وفك ترميز البث المتدرج `UTF-8` دون كسر الأحرف العربية ثنائية البايت.
- **النواقص:**
  1. عدم وجود دعم لـ `stop_sequences` (التوقف التلقائي عند توليد `<|user|>` أو `<|endoftext|>`).
  2. عدم وجود آلية إلغاء خارجي (`Cancel/Abort Token`) لإيقاف البث أثناء التوليد.

### 8.2 الخلل الحرج في تكامل الواجهة الأمامية مع الواجهة البرمجية (`UI-to-API Integration Bug`)
عند فحص الكود المصدري لكل من [api/backend.py](file:///d:/NourAI/api/backend.py#L324) و[ui/index.html](file:///d:/NourAI/ui/index.html#L222-L235):
1. **رفض الطلبات بـ `401 Unauthorized` في المتصفح الفعلي:**
   - يشترط `NourAIBackend.handle_request` وجود ترويسة `Authorization: Bearer <token>` لجميع المسارات باستثناء `/api/v1/health`, `/api/v1/info`, `/api/v1/auth/register`, `/api/v1/auth/token`.
   - في المقابل، ترسل دالة `sendMessage()` في [ui/index.html](file:///d:/NourAI/ui/index.html#L225-L229) طلب `POST /api/v1/chat/stream` بترويسة `{"Content-Type": "application/json"}` فقط دون أي توكن مصادقة!
   - النتيجة في المتصفح الحي: يعيد الخادم `401 Unauthorized`، وتكون `payload.stream_chunks` غير معرفة (`undefined`)، فتظهر فقاعة رد فارغة تماماً دون إظهار رسالة خطأ للمستخدم.
   - **لماذا لم تكشفه اختبارات المرحلة 19؟** لأن الاختبار في `test_phase19_ui.py` فحص نصوص `index.html` статيكياً واستدعى `NourAIBackend` مباشرة في بايثون بتوكن مُولَّد يدوياً!
2. **فقدان المحادثات عند إعادة التشغيل:**
   - يخزن `SessionManager` و`UserManager` في [api/backend.py](file:///d:/NourAI/api/backend.py) جميع البيانات في الذاكرة (`_sessions = {}`, `_users = {}`)، ولا يقرآن أو يكتبان في قاعدة البيانات الدائمة [nourai_production.sqlite3](file:///d:/NourAI/data/production/nourai_production.sqlite3).
3. **أزرار ووظائف مفقودة في واجهة المستخدم (`ui/index.html`):**
   - لا توجد واجهة تسجيل دخول أو تهيئة توكن تلقائية، ولا يوجد زر لحذف المحادثة، أو إعادة تسميتها، أو نسخ الإجابة، أو جلب قائمة الجلسات السابقة من `/api/v1/sessions`.

---

## المحور العاشر والحادي عشر والثاني عشر: الأمان، الإنتاج، الإصدارات، والصيانة (Axes 10, 11 & 12)

### 10.1 الأمان والخصوصية ([api/security_performance.py](file:///d:/NourAI/api/security_performance.py))
- **متحقق فعلياً:** تشفير كلمات المرور بـ `PBKDF2-HMAC-SHA256` (`100,000` تكرار مع `salt` عشوائي)، توكنات `HMAC-SHA256` موقعة، صلاحيات الأدوار (`admin`/`user`)، محدد معدل الطلبات (`RateLimiter`)، فلاتر حقن `SQLi`/`XSS`، وعزل جلسات المستخدمين.
- **النواقص:** غياب مسار حذف الحساب أو حذف المحادثات (`DELETE /api/v1/sessions/<id>`) لتمكين المستخدم من مسح بياناته.

### 11.1 النشر والإنتاج ([deployment/production.py](file:///d:/NourAI/deployment/production.py))
- **قاعدة البيانات والنسخ الاحتياطي:** قاعدة بيانات `SQLite` بوضع `WAL` مع النسخ الاحتياطي والتحقق من سلامة `PRAGMA integrity_check` تعمل فعلياً.
- **فجوة `TLS/HTTPS`:** دالة `start_live_http_server()` تشغل خادم `HTTP` عادي (`http://127.0.0.1:<port>`) دون تغليف المقبس بـ `ssl_context.wrap_socket()`، بينما تكتفي `build_tls_context()` بإنشاء كائن `ssl.SSLContext` في الذاكرة وفحص خاصية `minimum_version` فقط.

### 12.1 الإصدارات (`v1.0.0` و`v1.1.0`) والصيانة ([deployment/release_and_lifecycle.py](file:///d:/NourAI/deployment/release_and_lifecycle.py))
- **حزم الإصدارات:** ملفات [releases/v1.0.0/release_manifest.json](file:///d:/NourAI/releases/v1.0.0/release_manifest.json) و[releases/v1.1.0/release_manifest.json](file:///d:/NourAI/releases/v1.1.0/release_manifest.json) موجودة على القرص وبصمات `SHA-256` للأوزان مطابقة تماماً.
- **فجوة المحاكاة في المرحلة 23:** دالة `run_phase23_post_release_maintenance` تقوم بحقن حادثة وهمية مكتوبة مسبقاً (`TRANSIENT_TIMEOUT_WARN`) و`3` تقييمات ثابتة في قاعدة البيانات بدلاً من قراءة سجلات حية، كما لا توجد أداة `Rollback CLI` للتراجع الآلي من `v1.1.0` إلى `v1.0.0`.

---

## خطة الإصلاح والتنفيذ المرحلية المقترحة (Stages B → L)

| المرحلة القادمة | الأولوية | الإجراءات الهندسية المطلوبة لمعالجة الفجوات المكتشفة |
| :--- | :---: | :--- |
| **Stage B — Data Verification & Rebalancing** | **P0 (قصوى)** | تقسيم نصوص كتب `QNL` الطويلة إلى مقاطع متوازنة (`256–512` حرفاً)، وبناء تقسيمات متوازنة على أساس **عدد التوكنات (`33.3% AR / 33.3% EN / 33.3% FR`)** بدلاً من عدد السجلات، وإنشاء مجموعة تعليمات (`SFT`) حقيقية ومتنوعة خالية من التكرار والتسرب. |
| **Stage C — Model & Resume Verification** | **P1** | التأكد من دعم استئناف التدريب الكامل عبر `Trainer.resume(last.pt)` للحفاظ على حالة `AdamW` و`LambdaLR`، ومراجعة طول السياق ليتناسب مع البايتات العربية. |
| **Stage D & E — Real Training & Alignment** | **P0 (قصوى)** | تشغيل تدريب حقيقي كافٍ على البيانات المتوازنة تليه مرحلة ضبط تعليمات (`SFT`) حقيقية بدون تسرب بين التدريب والتقييم. |
| **Stage F, G & H — Multilingual, Capability & Safety Evaluation** | **P1** | إضافة مقاييس جودة التوليد الفعلي (صحة الكلمات، عدم التكرار `Distinct-N`، مطابقة لغة الإخراج للغة السؤال) إلى جانب `Perplexity`، وإعداد نموذج التقييم البشري. |
| **Stage I, J & K — Inference, UI/API & Production Hardening** | **P0 (قصوى)** | إضافة `stop_sequences` لمحرك الاستدلال؛ ربط `NourAIBackend` بقاعدة بيانات `SQLite` الدائمة؛ إصلاح مصادقة `ui/index.html` (`Bearer Token`) وإضافة أزرار حذف/تسمية/نسخ المحادثات؛ تفعيل `SSL/TLS` الاختياري الحقيقي وأداة `Rollback`. |
| **Stage L — Final Release `v1.2.0`** | **P1** | إصدار النسخة المعتمدة `v1.2.0` بعد اجتياز جميع البوابات الفنية والاختبارات الشاملة. |
