# المرحلة الأولى — تقرير التدقيق الجنائي الشامل لمشروع Jebrex AI

- **تاريخ التدقيق:** `2026-11-24`
- **المسار المفحوص:** `C:\Users\info\.gemini\antigravity\worktrees\NourAI\comprehensive_project_audit` (الفرع `comprehensive_project_audit` عند الالتزام `65fd2cd`)
- **اسم المشروع المستهدف:** `Jebrex AI` (الاسم التاريخي في المستودع والكود: `NourAI v1.2.0`)
- **بيئة التنفيذ المتحقَّق منها:** `D:\NourAI\.venv\Scripts\python.exe` (`Python 3.15.2`, `PyTorch 2.15.0+cpu`, `NumPy 2.5.4`, `pytest 9.2.0`)

---

## 1. جرد الموارد الفعلية المتاحة

| المورد | القيمة المقاسة فعلياً | الأثر الهندسي |
|---|---|---|
| نظام التشغيل | `Windows 10 Pro` | مدعوم محلياً |
| المعالج (CPU) | `Intel(R) Core(TM) i3-10110Y @ 1.00GHz` (`2` نواة فيزيائية / `4` خيوط، أقصى تردد `1.51 GHz`) | بطيء لتدريب النماذج الكبيرة؛ يكفي للنماذج المصغّرة واختبارات الاستدلال |
| الذاكرة العشوائية (RAM) | `7.89 GiB` إجمالي (`~2.51 GiB` متاحة عند الفحص) | يمنع تحميل أو تدريب نماذج تتجاوز بضع عشرات الملايين من المعاملات |
| البطاقة الرسومية (GPU / CUDA) | `Microsoft Basic Display Adapter` (`0 GiB VRAM`، `torch.cuda.is_available() == False`) | **ممنوع استخدام CUDA**؛ التدريب والاستدلال على `CPU FP32` فقط |
| التخزين `C:` (مسار العمل) | `6.95 GiB` فارغة من `~60 GiB` | حرج — يمنع نسخ بيانات ضخمة أو تثبيت حزم كبيرة |
| التخزين `D:` (المستودع الأصلي) | `14.70 GiB` فارغة | يحتوي على `D:\NourAI\.venv` وبيانات `raw/processed` المستثناة من Git (`~20.8 GB`) |
| أدوات التطبيقات | `tkinter` (متاح)، `java/javac` (JDK 25)، `adb` (متاح)، `gradle` (غير مثبت)، `fastapi/uvicorn` (غير مثبتين) | يمكن بناء وتشغيل تطبيق Windows (`tkinter`)، بينما تجميع APK لـ Android متوقف لغياب `gradle` |

---

## 2. تصنيف مكونات المشروع وفق الأدلة الفعلية

### أ. ما يعمل فعلياً (`Working & Verified`)
1. **المُرمِّز النصي `ByteTokenizer` (`tokenizer/byte_tokenizer.py`):** ترميز `UTF-8` غير فاقد (`vocab_size=260`: البايتات `0..255` والرموز الخاصة `PAD=256, BOS=257, EOS=258, UNK=259`)، يعمل بدقة `100%` عبر العربية والإنجليزية والفرنسية.
2. **معمارية `GPT` (`model/transformer.py`):** `Decoder-only Pre-LN Transformer` مع قناع سببي صحيح وربط أوزان التضمين برأس التوليد (`weight tying`) و3 أحجام معرفة (`default/nano=843,008`، `small=4,871,168`، `base=25,615,360` معاملاً).
3. **محرك التدريب `Trainer` و`CheckpointManager` (`training/trainer.py`, `training/checkpoint_manager.py`):** يدعم `AdamW`، `Warmup+Cosine`، تجميع التدرجات، قص التدرجات، الحفظ الذري مع `fsync`، وحفظ واستئناف حالة النموذج والمحسّن والمجدول و`RNG` و`EarlyStopping`.
4. **مجموعة الاختبارات الأساسية:** `250` اختباراً ناجحاً من أصل `262` (`1` متخطى، `11` فشلاً بسبب غياب ملفات `data/processed/phase10_3` و`data/raw` المستثناة في `.gitignore` داخل نسخة الـ Worktree).
5. **البنية التحتية للتخزين والإنتاج (`deployment/production.py`):** `SQLite WAL` مع النسخ الاحتياطي والاستعادة والتحقق بـ `PRAGMA integrity_check` وخادم `ThreadingHTTPServer` مع دعم `TLS 1.3`.

### ب. ما يعمل جزئياً (`Partially Working — Requires Hardening`)
1. **محرك الاستدلال (`inference/engine.py`):**
   - يعمل التوليد والبث (`stream_generate`) والإلغاء (`cancel_event`) والقياسات (`TTFT`, `tokens/sec`, `RAM`) بسرعة `~17.5–55.6` بايت/ثانية على CPU للنموذج `4.87M`.
   - **قصور مكتشف:** لا يدعم دمج `top_k` و`top_p` معاً؛ تسرّب `stop_sequences` متعددة البايتات أثناء البث التدريجي؛ لا يحقن تاريخ المحادثة متعدد الأدوار في `api/backend.py` (`formatted_prompt` يرسل آخر رسالة فقط!).
2. **الواجهة البرمجية (`api/backend.py`):**
   - مبنية على مكتبة بايثون القياسية (وليس `FastAPI`) وتدعم المصادقة `PBKDF2-HMAC-SHA256` و`HMAC-SHA256` و`SQLite` وتحديد المعدل.
   - **قصور مكتشف:** مواصفة OpenAPI ناقصة لا تغطي كل المسارات، غياب تمرير `client_ip` الحقيقي من الخادم في `deployment/production.py`، غياب حد حجم الطلب `Content-Length` ومعالجة `JSONDecodeError`، وغياب سجل تدقيق منقّح للأسرار.
3. **بيانات التدريب في مساحة العمل (`data/processed/v1_2_corpus` و`v1_2_sft`):**
   - خالية من التسرب الدقيق بين التقسيمات، ولكنها **ضئيلة جداً** (`134 KB` للـ corpus و`72` سطراً للـ SFT)، وتفتقر لحقول الترخيص داخل كل سجل.
   - **تلوث جودة حاد في العربية:** `31.9%` (`52/163`) من فقرات العربية في `v1_2_corpus/splits/train.jsonl` تحوي تشويش OCR حاداً وأحرفاً لاتينية عشوائية (`La que. Lame. ! Deum.`, `Lش خبص مبعنيناه`)، كما تحتوي بيانات `v1_2_sft` على أخطاء نحوية (`de le Qatar`، `يهدف الحوسبة`) وقوالب اصطناعية مكررة.
4. **واجهة الويب (`ui/index.html`):**
   - واجهة متجاوبة تدعم العربية `RTL` والإنجليزية والفرنسية وتعمل كـ `PWA`، لكنها تخلط افتراضياً بين النموذج العصبي وواجهات سحابية خارجية (`Gemini 2.0` و`Pollinations AI`) وتخزّن مفتاح `Gemini API` في Cookie غير محمي!

### ج. ما لا يعمل أو يفشل فعلياً (`Broken / Failing / Unsafe`)
1. **القدرة اللغوية التعميمية للنموذج العصبي (`checkpoints/v1_2/nourai_v1_2_model.pt` — `4.87M` معامل):**
   - أثبت الفحص المباشر أن النموذج يحفظ **6 عبارات مرجعية (`core_anchors`) حرفياً** لأن سكريبت `scripts/train_cloud_gpu.py` درّبه عليها بنسبة `50%–75%` من كل دفعة (تلوث تدريبي مباشر على عبارات الاختبار!).
   - عند اختباره على **أي سؤال غير مرئي** (حتى من `v1_2_sft/splits/test.jsonl` الخاص به مثل `"ما هي عاصمة الجزائر وما أهميتها؟"` أو أسئلة علوم/رياضيات/برمجة)، يُخرج **نصوصاً مشوّهة غير مفهومة** (`'أنا بخي مد ينة الدوحبي، هي اللمبكرريقللمركالثقاففي الئلثي ف'`، `'He is caramale astang Nepiapisiseceitens...'`).
2. **ثغرات أمنية حرجة (`CRITICAL`) في `api/backend.py`:**
   - **تنفيذ أوامر عشوائي (RCE) في `POST /api/v1/code/run`:** الاعتماد على قائمة حظر نصية ساذجة أمام `subprocess.run([sys.executable, "-I", "-c", code_str])` يتيح لأي ضيف تجاوزها بسهولة (`__import__('os')`, `pathlib`, إلخ).
   - **تصعيد صلاحيات (`Privilege Escalation`) في `POST /api/v1/users/register`:** قبول حقل `role` مباشرة من جسم الطلب العام يتيح لأي مستخدم غير مصادق إنشاء حساب `admin` فوراً!
   - **غياب حارس حقن الأوامر (`Prompt Injection`) في الوضع العصبي الافتراضي**، وغياب تحديد المعدل على `/api/v1/auth/token` و`/api/v1/users/register`، وغياب `HttpOnly` على الكوكي.
3. **11 اختباراً فاشلاً في مساحة العمل الحالية:** جميعها في `tests/test_phase10_*` و`tests/test_phase13_initial_training.py` بسبب الاعتماد على مسارات محلية مستثناة من Git (`data/raw/`, `data/processed/phase10_3/`, `data/metadata/*.sqlite3`).
4. **تعارض الوثائق والتقارير السابقة مع الواقع:**
   - تزعم شهادة `reports/final_audit/final_certification_v1_2_0.md` أن المشروع `FINAL_CERTIFIED` و`MULTILINGUAL_CAPABILITY_VERIFIED` بينما النموذج العصبي غير قادر على توليد جملة مفيدة واحدة لأسئلة غير مرئية.
   - تزعم `README.md` و`docs/ARCHITECTURE.md` أن النموذج النشط `843,008` معاملاً بينما `checkpoints/v1_2/nourai_v1_2_model.pt` و`releases/active_release.json` يحتويان على `4,871,168` معاملاً (`Small`, `context_length=256`).

### د. ما لم يتم بناؤه أو اختباره (`Missing / Untested`)
- **منظومة إدارة المشروع الكاملة:** غياب ملفات `Master Roadmap`, `Project Dashboard`, `Task Tracker`, `Evidence Registry`, `Risk Register`, `Blockers Register`, `Decision Log`, `Resource Plan`.
- **تطبيق سطح المكتب لـ Windows:** لا يوجد تطبيق مكتبي مستقل (يوجد فقط مشغّل طرفي/متصفح `run_nourai.py`).
- **تطبيق Android الأصلي:** لا يوجد مشروع Kotlin/Android أصلي (يوجد فقط `ui/manifest.json` لـ PWA).
- **اختبار استقلالية النموذج ومعيار مقارن موحد مع ChatGPT وGoogle Gemini:** غير موجود.
- **التقييم البشري الموثّق:** غير موجود.

### هـ. ما يجب الاحتفاظ به مقابل ما يحتاج إلى معالجة/إعادة بناء
- **يُحتفظ به دون هدم:** `tokenizer/byte_tokenizer.py`، `model/transformer.py` (مع تحسين طفيف متوافق عكسياً مثل `ignore_index`)، `training/trainer.py`، `training/checkpoint_manager.py`، `deployment/production.py` (مع التحصين الأمني)، جميع نقاط الحفظ والتقارير والبيانات السابقة (دون حذف أي منها).
- **يُعاد بناؤه/يُحصَّن فوراً:** بوابة جاهزية البيانات المستقلة لكل لغة (`data/jebrex_data_gate.py`) وتنقية البيانات دون تلوث؛ سياق المحادثة وبث `stop_sequences` ودمج `top_k/top_p` في `inference/engine.py`؛ إصلاح الثغرات الحرجة في `api/backend.py` و`deployment/production.py` و`ui/index.html`؛ فصل النموذج العصبي المستقل تماماً عن أي مزود خارجي؛ منظومة تقييم ومعيار مقارن موحد لـ 12 مجالاً؛ تطبيق Windows وبنية مشروع Android.

### و. العوائق التي تمنع الوصول لمستوى ChatGPT/Gemini حالياً
1. **عتاد الجهاز المحلي:** معالج `i3-10110Y` ثنائي النواة بلا GPU وبذاكرة حرة `~2.5 GB` وتخزين حر `6.95 GB` لا يسمح بتدريب نموذج يتجاوز ملايين قليلة من المعاملات.
2. **حجم وجودة البيانات الموثّقة:** للوصول إلى قدرات محادثة واستدلال عامة يلزم عشرات إلى مئات المليارات من التوكنات النظيفة المرخّصة ومُرمِّز `Subword BPE` (وليس `134 KB` بترميز بايتات خام مع `context_length=256`).
