---
title: "NourAI v3.2.0 — الخريطة الذهنية والنظام المعرفي الشامل"
aliases:
  - NourAI Knowledge Map
  - الخريطة المعرفية لمشروع NourAI v3.2
tags:
  - nourai
  - knowledge-engineering
  - mindmap
  - moc
  - v3-2-0
version: "3.2.0"
updated: "2026-11-20"
compatible_with:
  - Obsidian
  - Logseq
  - Freeplane (OPML)
  - TheBrain
---

# 🧠 [[NourAI]] `v1.2.0` — الخريطة الذهنية والنظام المعرفي الشامل

> [!IMPORTANT]
> **الفكرة المركزية (Central Thought):**
> بناء نظام ذكاء اصطناعي لغوي ومعرفي **محلي أولاً (`Local-First`)** ومتعدد اللغات (**العربية · الإنجليزية · الفرنسية**) يعمل بأقل اعتماديات ممكنة (مكتبة بايثون القياسية + `PyTorch` + `NumPy`)، ويجمع بين نموذج `Decoder-only Pre-LN Transformer` وبنية معرفية من 8 طبقات، وذاكرة طويلة المدى، و`RAG` محلي، وبحث ويب حي آمن، ومعايير إنتاجية كاملة (`TLS 1.3`, `SQLite WAL`, `SHA-256 Provenance`, `Rollback`).

---

## 🎨 1. نظام الألوان والرموز والأولويات (Visual Taxonomy & Legend)

يعتمد هذا النظام المعرفي على ترميز موحّد قابل للقراءة البشرية والبرمجية في **Obsidian** (عبر Callouts والوسوم) و**Logseq** (عبر خصائص الكتل `::`) و**Freeplane** (عبر الأيقونات والألوان):

| النوع المعرفي | الرمز | اللون الدلالي | الوسم `#tag` | الاستخدام |
| :--- | :---: | :--- | :--- | :--- |
| **الفكرة المركزية / المفهوم الجوهري** | 🧠 | أزرق ملكي (`#1E40AF`) | `#core` | العقد المحورية والمعمارية الأساسية |
| **هدف استراتيجي (Goal)** | 🎯 | بنفسجي (`#6D28D9`) | `#goal` | النتائج بعيدة ومتوسطة المدى |
| **مشروع / مسار عمل (Project)** | 📦 | أزرق سماوي (`#0284C7`) | `#project` | حزم العمل القابلة للتسليم والإصدار |
| **مهمة تنفيذية (Task)** | ✅ | أخضر زمردي (`#059669`) | `#task` | خطوات عملية محددة وقابلة للتحقق |
| **فكرة / ابتكار (Idea)** | 💡 | ذهبي (`#D97706`) | `#idea` | مقترحات تطويرية وابتكارات مستقبلية |
| **مرجع / توثيق / أصل (Reference)** | 📖 | رمادي أردوازي (`#475569`) | `#ref` | ملفات الكود، التقارير، المعايير، والبيانات |
| **ثغرة / مخاطرة / نقص (Gap/Risk)** | ⚠️ | أحمر قرمزي (`#DC2626`) | `#gap` | نقاط الضعف، القيود، والمعلومات الناقصة |
| **رابط ترابطي عابر للفروع (Jump Link)** | 🔗 | فيروزي (`#0D9488`) | `#link` | علاقات `TheBrain` الشبكية غير الهرمية |

### مستويات الأولوية والأهمية (Priority Classification)
- **🔴 `P0` — حرج (Critical):** يؤثر مباشرة على صحة النموذج أو الأمان أو استقرار الإنتاج.
- **🟠 `P1` — عالٍ (High):** ضروري لتحقيق القيمة الأساسية وتوسيع القدرات اللغوية.
- **🟡 `P2` — متوسط (Medium):** تحسينات مهمة لتجربة المستخدم والكفاءة والتوثيق.
- **🟢 `P3` — توسعي (Low / Exploratory):** ابتكارات مستقبلية ومزايا الجيل القادم.

---

## 🌳 2. الهيكل الهرمي الشامل للخريطة الذهنية (Logseq Outliner & Obsidian MOC)

> [!TIP]
> كُتب هذا القسم بتنسيق **Outliner متداخل** مع خصائص كتل Logseq (`type::`, `priority::`, `color::`) وروابط ثنائية الاتجاه لـ Obsidian (`[[...]]`) ليعمل مباشرة في كلا الأداتين دون تعديل.

- 🧠 **[[NourAI v1.2.0]] — المركز المعرفي للمشروع**
  type:: core
  priority:: P0
  color:: #1E40AF
  - 🎯 **الفرع 1: الهوية والرؤية والأهداف الاستراتيجية (`[[Strategic Vision & Goals]]`)**
    type:: goal
    priority:: P0
    color:: #6D28D9
    - 🎯 **الاستقلالية المحلية (`[[Local-First Sovereignty]]`)**
      - تشغيل كامل دون خوادم خارجية إجبارية ودون اعتماديات ثقيلة (`FastAPI`/`uvicorn` مستبدلة بـ `ThreadingHTTPServer`).
      - 🔗 يرتبط بـ: [[API Backend]]، [[Production Deployment]]، [[Hardware Constraints]].
    - 🎯 **التكافؤ اللغوي الثلاثي (`[[Multilingual Parity AR-EN-FR]]`)**
      - توازن دقيق بنسبة `~33.3%` لكل لغة في المدونة النصية والتقييم العادل لكل لغة على حدة.
      - 🔗 يرتبط بـ: [[ByteTokenizer]]، [[Balanced Corpus v1.2]]، [[Multilingual Evaluation]].
    - 🎯 **الشفافية الهندسية الصادقة (`[[Engineering Honesty & Telemetry]]`)**
      - فصل صريح بين `response_source="neural_model"` و`response_source="rule_based_fallback_assistant"` وعدم تلفيق مقاييس عصبية.
      - 🔗 يرتبط بـ: [[Inference Engine]]، [[Fallback Assistant]]، [[Gap: Nano Model Coherency]].
    - 🎯 **الجاهزية الإنتاجية والقابلية للاسترجاع (`[[Production Readiness & Reproducibility]]`)**
      - توثيق `SHA-256` من المصدر حتى نقطة الحفظ (`Checkpoint`)، مع آلية تراجع آلي (`Rollback`) بين الإصدارات `v1.0.0 ↔ v1.1.0 ↔ v1.2.0`.
      - 🔗 يرتبط بـ: [[Release Lifecycle]]، [[Data Provenance]]، [[Security Architecture]].

  - 🧠 **الفرع 2: المعمارية العصبية للنموذج (`[[Neural Model Architecture]]`)**
    type:: core
    priority:: P0
    color:: #1E40AF
    - 📦 **المُرمِّز النصي (`[[ByteTokenizer]]`)**
      ref:: `tokenizer/byte_tokenizer.py`
      priority:: P0
      - ترميز `UTF-8` غير فاقد (`Lossless`) بحجم مفردات `vocab_size = 260`.
      - البايتات الخام `0..255` + أربعة رموز خاصة: `256:<PAD>`, `257:<BOS>`, `258:<EOS>`, `259:<UNK>`.
      - ⚠️ ثغرة مرتبطة: [[Gap: Arabic Byte-Level Context Tax]] (الحرف العربي يستهلك 2 بايت مما يقلّص النافذة الفعلية).
    - 📦 **العمود الفقري المحوّل (`[[Pre-LN Decoder-Only Transformer]]`)**
      ref:: `model/transformer.py`, `model/factory.py`, `model/architecture_audit.py`
      priority:: P0
      - معمارية `GPT` بمحاذاة `Pre-LayerNorm` وربط أوزان التضمين برأس الإخراج (`Weight Tying: head.weight is token.weight`).
      - **الأحجام القياسية (`[[Model Presets]]`):**
        - `Default / Nano`: `4` طبقات · `4` رؤوس · `d_model=128` · `context=128` ← **`843,008` معامل** (المُدرَّب حالياً في `v1.2.0`).
        - `Small`: `6` طبقات · `8` رؤوس · `d_model=256` · `context=256` ← **`4,871,168` معامل**.
        - `Base`: `8` طبقات · `8` رؤوس · `d_model=512` · `context=512` ← **`25,615,360` معامل**.
      - 🔗 يرتبط بـ: [[Training Engine]]، [[Cloud GPU Scaling]]، [[Checkpoint Format]].

  - ⚙️ **الفرع 3: البنية المعرفية ذات الثماني طبقات (`[[8-Layer Cognitive Architecture]]`)**
    type:: core
    priority:: P1
    color:: #1E40AF
    ref:: `nourai/engines.py`
    - 📦 **الطبقة 1 — `[[FoundationModelEngine]]`**: تشغيل النموذج المحلي (`PyTorch`) مع إمكانية التحويل الاحتياطي.
    - 📦 **الطبقة 2 — `[[ReasoningEngine]]`**: تصنيف النوايا (`Intent Classification`) وبناء مسار التفكير (`ReasoningTrace`).
    - 📦 **الطبقة 3 — `[[MemoryEngine]]`**: استخراج حقائق المستخدم وتذكرها طويل المدى (`nourai/memory_and_rag.py`).
    - 📦 **الطبقة 4 — `[[KnowledgeEngine]]`**: استرجاع المستندات المحلية (`Zero-Bloat SQLite RAG`) لملفات `.docx` و`.pdf` و`.txt` عبر مكتبة بايثون القياسية فقط.
    - 📦 **الطبقة 5 — `[[CodingEngine]]`**: اكتشاف الأكواد، تنسيقها، وتقييم التعبيرات الحسابية بأمان عبر `AST` (`_safe_eval_ast`) دون `eval/exec`.
    - 📦 **الطبقة 6 — `[[EvaluationEngine]]`**: التحقق الذاتي من جودة الإجابة قبل عرضها.
    - 📦 **الطبقة 7 — `[[AgentEngine]]`**: اختيار الأدوات تلقائياً وتنفيذها (حساب، بحث، مستندات، برمجة).
    - 📦 **الطبقة 8 — `[[NourAIPlatform]]`**: المنسّق الأعلى الذي يربط الطبقات السبع بواجهة المحادثة والـ API.
    - 🔗 يرتبط بـ: [[Live Web Search]]، [[Fallback Assistant]]، [[Web UI & Canvas]].

  - 📚 **الفرع 4: هندسة البيانات والمدونات النصية (`[[Data & Corpus Engineering]]`)**
    type:: project
    priority:: P0
    color:: #0284C7
    ref:: `corpus/`, `data/`, `datasets/`
    - 📦 **الاستحواذ والمحوّلات (`[[Corpus Acquisition & Adapters]]`)**
      - محوّلات مخصصة: مكتبة قطر الوطنية (`corpus/adapters/qnl_adapter.py`) ومنظمة التجارة العالمية (`corpus/adapters/wto_adapter.py`) ومجموعة `Tatoeba CC0`.
      - سجل مصادر وتراخيص صارم (`data/license_policy.py`, `corpus/licensing.py`).
    - 📦 **النزاهة ومنع التسرب (`[[Data Provenance & Leakage Audit]]`)**
      - بصمات `SHA-256` لكل ملف خام ومُعالج (`data/provenance.py`, `corpus/integrity.py`).
      - إزالة التكرار الدقيق وشبه الدقيق (`data/deduplication.py`, `data/near_duplicates.py`).
      - تدقيق صارم لعدم تسرب المستندات أو مجموعات الترجمة بين التقسيمات (`corpus/leakage_audit.py` ← `0 Leakage`).
    - 📦 **المدونات المعتمدة (`[[Balanced Corpus v1.2]]` & `[[SFT Dataset v1.2]]`)**
      - `v1_2_corpus`: تقسيمات `train/val/test` متوازنة بالتوكنات/البايتات (`~33.3%` لكل من AR/EN/FR).
      - `v1_2_sft`: مجموعة تعليمات بدون تداخل (`72` تدريب / `36` تحقق / `36` اختبار).
      - ⚠️ ثغرة مرتبطة: [[Gap: Small SFT Dataset Size]].
    - 📦 **محمّل البيانات والتخزين (`[[Dataset Loader & Metadata Catalog]]`)**
      - فهرس بيانات وصفية في `SQLite` (`data/metadata/corpus.sqlite3`) وذاكرة تخزين مؤقت للتوكنات (`token_cache`).

  - 🚀 **الفرع 5: محرك التدريب والتحسين والمواءمة (`[[Training & Alignment Pipeline]]`)**
    type:: project
    priority:: P0
    color:: #0284C7
    ref:: `training/`, `scripts/train_and_evaluate_neural_v1_2.py`, `scripts/train_cloud_gpu.py`
    - 📦 **محرك التدريب الأساسي (`[[Training Engine]]`)**
      - إدارة الحلقات (`trainer.py`)، المحسّن (`optimizer.py`)، المجدول (`scheduler.py`)، الإيقاف المبكر (`early_stopping.py`)، وإدارة نقاط الحفظ (`checkpoint_manager.py`).
      - سجل أحداث `JSONL` ومراقبة استهلاك الذاكرة (`logger.py`, `metrics.py`).
    - 📦 **مراحل التدريب المنفذة (`[[Pretraining & Response-Masked SFT]]`)**
      - تدريب أولي (`150` خطوة: هبوط الخسارة من `5.5773` إلى `2.2990`).
      - ضبط دقيق مقنّع للاستجابة (`400` خطوة: هبوط الخسارة من `2.9181` إلى `1.3016`) في `502.76` ثانية على المعالج المركزي.
      - تدريب تحسين مستمر (`Phase 24`: `reports/phase24/`).
    - 📦 **التوسع السحابي (`[[Cloud GPU Scaling]]`)**
      - دفتر `Colab/Kaggle` (`notebooks/NourAI_Cloud_GPU_Training_Colab_Kaggle.ipynb`) وسكربت `scripts/train_cloud_gpu.py`.
      - ⚠️ ثغرة مرتبطة: [[Gap: CPU-Only Hardware Bottleneck]] و[[Gap: Unverified CUDA AMP/OOM]].

  - ⚡ **الفرع 6: محرك الاستدلال والقدرات الحية (`[[Inference & Hybrid Capabilities]]`)**
    type:: project
    priority:: P1
    color:: #0284C7
    ref:: `inference/engine.py`, `nourai/capabilities.py`, `nourai/web_search.py`, `nourai/memory_and_rag.py`
    - 📦 **الاستدلال العصبي المباشر (`[[Inference Engine]]`)**
      - الوضع الافتراضي (`use_capabilities=False`)، توليد تدفقي (`Streaming`)، قياس زمن الاستجابة وعدد التوكنات الحقيقي.
    - 📦 **المساعد القاعدي الاحتياطي (`[[Fallback Assistant]]`)**
      - وضع اختياري (`use_fallback_assistant=True`) يوفّر ردوداً مهيكلة مع الإفصاح الصريح عن المصدر.
    - 📦 **البحث الحي المحمي (`[[Live Web Search]]`)**
      - مصادر مجانية: `Wikipedia + Google News RSS + DuckDuckGo`.
      - حماية أمنية ضد `SSRF` عبر `validate_public_url` وحظر العناوين الخاصة (`localhost`, `metadata.google.internal`, النطاقات غير العمومية).
    - 📦 **الذاكرة و`RAG` المحلي (`[[Local Memory & Document RAG]]`)**
      - حفظ حقائق المستخدم في `SQLite`، واستخلاص نصوص `.docx/.pdf/.txt` بحد أقصى `5 MB` للملف و`10 MB` لفك ضغط XML (حماية ضد Zip Bombs).

  - 🌐 **الفرع 7: الواجهة البرمجية وتجربة المستخدم (`[[API & User Experience]]`)**
    type:: project
    priority:: P1
    color:: #0284C7
    ref:: `api/backend.py`, `ui/index.html`, `ui/chat_app.py`, `run_nourai.py`
    - 📦 **خادم الواجهة البرمجية (`[[API Backend]]`)**
      - مبني على `ThreadingHTTPServer` + `ssl` بوثيقة `OpenAPI 3.1.0` (`/api/v1/openapi.json`).
      - نقاط النهاية: `/health`, `/users/register`, `/auth/token`, `/auth/guest`, `/sessions`, `/chat`, `/chat/stream`, `/admin/users`.
      - ⚠️ ثغرة مرتبطة: [[Gap: Outdated API Reference Doc]].
    - 📦 **واجهة الويب والطرفية (`[[Web UI & Canvas]]`)**
      - واجهة متجاوبة `RTL/LTR` (`ui/index.html`) تدعم العربية والإنجليزية والفرنسية وتعرض شارة مصدر الرد وتسمح بالتبديل الصريح.
      - معاينة حية للأكواد (`Live HTML/JS Canvas`)، وملفات فهرسة وتطبيق ويب (`manifest.json`, `llms.txt`, `ai-catalog.json`, `sitemap.xml`).
      - واجهة طرفية تفاعلية (`run_nourai.py --cli`).

  - 🛡️ **الفرع 8: الأمان والإنتاج وإدارة الإصدارات (`[[Security, Production & DevOps]]`)**
    type:: project
    priority:: P0
    color:: #0284C7
    ref:: `api/security_performance.py`, `security/audit.py`, `deployment/production.py`, `deployment/release_and_lifecycle.py`
    - 📦 **منظومة الأمان (`[[Security Architecture]]`)**
      - تجزئة كلمات المرور `PBKDF2-HMAC-SHA256` (`100,000` تكرار).
      - رموز مصادقة `HMAC-SHA256` بمفتاح من `NOURAI_SECRET_KEY` (أو توليد عشوائي آمن عبر `secrets.token_hex(32)`).
      - تحديد معدل الطلبات (`Rate Limiting`) وصلاحيات الأدوار (`RBAC`) وحماية `SSRF` ورؤوس أمان HTTP (`X-Frame-Options`, `nosniff`, `Permissions-Policy`).
    - 📦 **النشر والتشغيل الإنتاجي (`[[Production Deployment]]`)**
      - خادم `HTTPS (TLSv1.3)`، قاعدة بيانات `SQLite WAL` مع نسخ احتياطي موثّق بـ `SHA-256`.
      - حاوية `Dockerfile` غير جذرية (`appuser:10001`, منفذ `7860`)، نشر سحابي عبر `render.yaml` و`firebase.json`، وتكامل مستمر عبر `.github/workflows/ci.yml`.
    - 📦 **دورة حياة الإصدارات (`[[Release Lifecycle]]`)**
      - سجل إصدارات موثّق: `v1.0.0` → `v1.1.0` → `v1.2.0` مع `SHA256SUMS.txt`، `rollback_plan.json`، وسكربت تراجع آلي (`scripts/rollback_release.py`).

  - 📊 **الفرع 9: التقييم والاختبار وضمان الجودة (`[[Evaluation & QA System]]`)**
    type:: project
    priority:: P1
    color:: #0284C7
    ref:: `evaluation/`, `tests/`, `reports/`
    - 📦 **محرك التقييم اللغوي (`[[Multilingual Evaluation]]`)**
      - قياس الخسارة (`Loss`)، الحيرة اللغوية (`Perplexity`)، والبت لكل بايت (`Bits-per-Byte`) منفصلةً لكل لغة (`ar`, `en`, `fr`).
      - مقارنة التوليد قبل التدريب وبعده (`generation_before` vs `generation_after`).
    - 📦 **حزمة الاختبارات والتدقيق (`[[Test Suite & Phase Audits]]`)**
      - اختبارات `pytest` تغطي المراحل 8 و9 و10 وشهادات المراحل `B` إلى `L`.
      - تقارير تدقيق ممتدة عبر 24 مرحلة هندسية (`reports/phase10` إلى `reports/phase24` + تقارير الجذر `PHASE_3`–`PHASE_10_1`).

---

## 🧩 3. الفصل الصريح للكيانات المعرفية (Separation of Entities)

### 🎯 أ. الأهداف (`#goal`)
1. **[P0 🔴]** نقل جودة التوليد العصبي من مرحلة "إثبات البنية التحتية (`Nano 843K`)" إلى "توليد لغوي متماسك متعدد اللغات (`Base 25M+` أو `Small 4.8M`)" عبر التدريب السحابي على GPU. ← [[Cloud GPU Scaling]]
2. **[P0 🔴]** تقليص ضريبة طول السياق للغة العربية عبر الانتقال أو التهجين مع مُرمِّز فرعي الكلمات (`BPE/Unigram`) دون فقدان التغطية الشاملة لـ UTF-8. ← [[ByteTokenizer]]
3. **[P1 🟠]** توسيع مجموعة بيانات الضبط الدقيق (`SFT`) من `72` مثالاً إلى `+5,000` مثال متوازن عالي الجودة (عربي/إنجليزي/فرنسي) خالٍ من التسرب. ← [[SFT Dataset v1.2]]
4. **[P2 🟡]** تحقيق توثيق متزامن 100% مع الكود الحي (`v1.2.0+`) وإطلاق لوحة قياس أداء تفاعلية. ← [[Gap: Outdated API Reference Doc]]

### 📦 ب. المشاريع الفرعية النشطة (`#project`)
1. **مشروع الترقية العصبية `v1.3.0 / v2.0.0`**: تدريب `Small (4.8M)` ثم `Base (25.6M)` على GPU سحابي وتحديث `releases/active_release.json`.
2. **مشروع توسيع المدونة و`SFT`**: بناء خط أنابيب بيانات تعليمية متوازنة للغات الثلاث مع تدقيق التسرب.
3. **مشروع ترقية المحرك المعرفي (`RAG 2.0`)**: تطوير البحث الدلالي (`Hybrid BM25 + Embeddings`) وتحسين استخلاص ملفات PDF العربية.
4. **مشروع توثيق وتوحيد المنصة**: تحديث `docs/API_REFERENCE.md` وتنظيف ملفات الاختبار المؤقتة في `reports/phase10/pytest-temp`.

### ✅ ج. المهام التنفيذية المصنفة بالأولوية (`#task`)
- [ ] **[P0 🔴]** تشغيل تدريب نموذج `Small` أو `Base` عبر `scripts/train_cloud_gpu.py` على GPU (Colab/Kaggle) والتحقق من دعم `CUDA AMP` والتعافي من `OOM`. #task #training
- [ ] **[P0 🔴]** تشخيص سبب ارتفاع حيرة اللغة العربية بعد SFT (`6.1365 → 7.8313`) وإعادة ضبط معدل التعلم وعدد خطوات `SFT` لمنع النسيان الكارثي (`Catastrophic Forgetting`). #task #evaluation
- [ ] **[P1 🟠]** توسيع `data/processed/v1_2_sft` من `144` مثالاً إجمالياً إلى مجموعة غنية متعددة المهام (محادثة، تلخيص، ترجمة، برمجة، استدلال) مع الحفاظ على `0 Leakage`. #task #data
- [ ] **[P1 🟠]** تصميم وتجربة مُرمِّز `Byte-Level BPE` (مفردات `8K–32K`) ومقارنة `tokens/word` للعربية مقابل `ByteTokenizer (260)`. #task #tokenizer
- [ ] **[P2 🟡]** تحديث [API_REFERENCE.md](file:///d:/NourAI/docs/API_REFERENCE.md) من `v1.0.0` إلى `v1.2.0` لتضمين `/api/v1/auth/guest`، خيارات `use_fallback_assistant`، ومسارات الذاكرة و`RAG`. #task #docs
- [ ] **[P2 🟡]** تحسين محلّل PDF في [memory_and_rag.py](file:///d:/NourAI/nourai/memory_and_rag.py) لدعم الخطوط العربية وترميز `CMap / ToUnicode` بدلاً من استخراج السلاسل بين الأقواس فقط. #task #rag
- [ ] **[P3 🟢]** إضافة أتمتة تنظيف لمجلدات `reports/phase10/pytest-temp` ومنع تراكمها داخل المستودع. #task #devops

### 💡 د. الأفكار والابتكارات المستقبلية (`#idea`)
- 💡 **[P1 🟠] `[[Byte-Level BPE Multilingual Tokenizer]]`**: الاحتفاظ بميزة عدم وجود `<UNK>` مع مضاعفة السياق الفعلي للنص العربي `4x–6x`.
- 💡 **[P1 🟠] `[[Hybrid Local RAG (BM25 + Tiny Embeddings)]]`:**: دمج البحث المعجمي `SQLite FTS5` مع متجهات خفيفة محلية لتحسين دقة الاسترجاع دون كسر فلسفة `Zero-Bloat`.
- 💡 **[P2 🟡] `[[INT8/INT4 CPU Quantization & KV-Cache]]`**: تكميم النموذج `Base (25.6M)` وتفعيل `KV-Cache` لتشغيل استدلال سريع وسلس على أجهزة CPU محدودة مثل `Core i3 / 8GB RAM`.
- 💡 **[P2 🟡] `[[Graph-Aware Long-Term Memory]]`**: تحويل الذاكرة الدائمة من قائمة حقائق مسطحة في `SQLite` إلى رسم بياني معرفي شخصي (`Personal Knowledge Graph`) متوافق مع Obsidian.
- 💡 **[P2 🟡] `[[GGUF / ONNX Export Bridge]]`**: تصدير نقاط حفظ NourAI إلى صيغ قياسية لتسهيل التشغيل على الهواتف والمتصفح (`WebGPU / WebAssembly`).
- 💡 **[P3 🟢] `[[Voice RTL/LTR Local Interface]]`**: استثمار تصريح `microphone=(self)` الموجود مسبقاً في `firebase.json` لإضافة تفاعل صوتي محلي (STT/TTS) باللغات الثلاث.
- 💡 **[P3 🟢] `[[Federated / LoRA Personal Adapters]]`**: تمكين كل مستخدم من تدريب محوّل `LoRA` خفيف محلياً على مستنداته دون تعديل أوزان النموذج الأساسي.

### 📖 هـ. المراجع والأصول الأساسية (`#ref`)
- 📖 [README.md](file:///d:/NourAI/README.md) — النظرة العامة وأوامر التشغيل والاختبار.
- 📖 [ARCHITECTURE.md](file:///d:/NourAI/docs/ARCHITECTURE.md) — المواصفات المعمارية للنموذج والـ API.
- 📖 [API_REFERENCE.md](file:///d:/NourAI/docs/API_REFERENCE.md) — مرجع نقاط النهاية (يحتاج تحديثاً لـ `v1.2.0`).
- 📖 [HARDWARE_CAPABILITIES.md](file:///d:/NourAI/HARDWARE_CAPABILITIES.md) — توصيف قدرات العتاد المحلي (`i3-10110Y`, `7.9GB RAM`, `CPU-only`).
- 📖 [evaluation_report.json](file:///d:/NourAI/releases/v1.2.0/evaluation_report.json) — النتائج الرقمية ونماذج التوليد الحقيقية لإصدار `v1.2.0`.
- 📖 [CHECKPOINT_FORMAT.md](file:///d:/NourAI/CHECKPOINT_FORMAT.md) و[TRAINING_ENGINE_GUIDE.md](file:///d:/NourAI/TRAINING_ENGINE_GUIDE.md) — معايير نقاط الحفظ ومحرك التدريب.

---

## ⚠️ 4. تحليل الثغرات والمعلومات الناقصة (`[[Gaps & Missing Information]]`)

كشف الفحص العميق للمشروع عن **7 ثغرات حقيقية** يجب معالجتها لضمان انتقال NourAI من بنية هندسية متينة إلى نموذج لغوي فائق القدرة:

| المعرّف | الثغرة / النقص المكتشف | الشدة | الدليل من المشروع | الأثر |
| :--- | :--- | :---: | :--- | :--- |
| **`[[Gap: Nano Model Coherency]]`** | عدم تماسك التوليد العصبي المفتوح حالياً | 🔴 `P0` | `releases/v1.2.0/evaluation_report.json` (`generation_after` يُنتج نصوصاً غير مفهومة في الأسئلة المفتوحة) | يضطر المستخدم للاعتماد على المساعد القاعدي الاحتياطي بدلاً من الشبكة العصبية. |
| **`[[Gap: Arabic Perplexity Regression]]`** | تراجع حيرة اللغة العربية بعد SFT | 🔴 `P0` | `ar perplexity`: ارتفعت من `6.1365` قبل SFT إلى `7.8313` بعده | مؤشر على فرط التخصيص (`Overfitting`) أو النسيان الكارثي بسبب صغر بيانات SFT. |
| **`[[Gap: Arabic Byte-Level Context Tax]]`** | كفاءة السياق المنخفضة للعربية في `ByteTokenizer` | 🔴 `P0` | `vocab_size=260` و`context_length=128` ← ~60 حرفاً عربياً فقط! | يستحيل على النموذج فهم محادثة متعددة الأدوار أو سياق RAG ضمن 128 بايت. |
| **`[[Gap: Small SFT Dataset Size]]`** | صغر حجم مجموعة بيانات التعليمات (`SFT`) | 🟠 `P1` | `72` تدريب / `36` تحقق / `36` اختبار في `v1_2_sft` | لا يكفي لتعميم اتباع التعليمات باللغات الثلاث. |
| **`[[Gap: CPU-Only Hardware Bottleneck]]`** | قيود العتاد المحلي وعدم اختبار مسار GPU | 🟠 `P1` | `HARDWARE_CAPABILITIES.md`: `i3-10110Y`, `CPU FP32 only`, `CUDA AMP/OOM unverified` | يحدّ من تدريب `Small/Base` محلياً ويجعل مسار السحابة بحاجة لاختبار تحقّق فعلي. |
| **`[[Gap: Outdated API Reference Doc]]`** | عدم مزامنة توثيق الـ API مع الإصدار `v1.2.0` | 🟡 `P2` | `docs/API_REFERENCE.md` ما زال بعنوان `v1.0.0` ويفتقد `/auth/guest` و`RAG/Memory` | فجوة بين التوثيق والواقع البرمجي للمطورين الجدد. |
| **`[[Gap: Arabic PDF Parsing]]`** | محدودية استخلاص النصوص العربية من PDF | 🟡 `P2` | `nourai/memory_and_rag.py` يعتمد على Regex للأقواس داخل `FlateDecode` | يفشل مع معظم ملفات PDF العربية المشفّرة بجداول `ToUnicode CMap` أو الخطوط المدمجة. |

---

## 🔗 5. مصفوفة الروابط الثنائية والعلاقات المنطقية (`[[Cross-Domain Relationships]]`)

> علاقات صريحة ثنائية الاتجاه تربط بين الفروع دون تكرار المعلومات (للتفصيل الكامل بأسلوب TheBrain راجع [[TheBrain_Network_Map]]):

- `[[ByteTokenizer]]` ⇄ **`يؤثر سلباً على السياق في`** ⇄ `[[Gap: Arabic Byte-Level Context Tax]]` ⇄ **`يُعالج بواسطة`** ⇄ `[[Byte-Level BPE Multilingual Tokenizer]]`
- `[[Pre-LN Decoder-Only Transformer]]` ⇄ **`يُدرَّب عبر`** ⇄ `[[Training Engine]]` ⇄ **`يحتاج لتجاوز قيود`** ⇄ `[[Gap: CPU-Only Hardware Bottleneck]]` ⇄ **`عبر`** ⇄ `[[Cloud GPU Scaling]]`
- `[[Balanced Corpus v1.2]]` + `[[SFT Dataset v1.2]]` ⇄ **`يُقيَّم عبر`** ⇄ `[[Multilingual Evaluation]]` ⇄ **`كشف عن`** ⇄ `[[Gap: Arabic Perplexity Regression]]`
- `[[Inference Engine]]` ⇄ **`يتكامل بشفافية مع`** ⇄ `[[Fallback Assistant]]` ⇄ **`يُعرض مصدره في`** ⇄ `[[Web UI & Canvas]]`
- `[[8-Layer Cognitive Architecture]]` ⇄ **`يستثمر`** ⇄ `[[Local Memory & Document RAG]]` + `[[Live Web Search]]` ⇄ **`محمية بواسطة`** ⇄ `[[Security Architecture]]`
- `[[Data Provenance & Leakage Audit]]` ⇄ **`يضمن نزاهة`** ⇄ `[[Release Lifecycle]]` ⇄ **`يُمكّن`** ⇄ `[[Production Deployment]]`

---

## 🔄 6. مقترحات تطوير النظام المعرفي نفسه مستقبلاً (Evolving the Knowledge System)

لضمان بقاء هذه الخريطة حية ومتنامية مع تطور NourAI من `v1.2.0` إلى `v2.0.0+`:

1. **التكامل الآلي مع الكود (`Docs-as-Knowledge-Graph`):**
   - إضافة سكربت بايثون خفيف (`scripts/sync_knowledge_map.py`) يقرأ `releases/active_release.json` وتقارير `evaluation_report.json` ليُحدّث تلقائياً أرقام الخسارة والحيرة وحالة الإصدار داخل ملفات Markdown وOPML.
2. **تقسيم الذرات المعرفية (`Atomic Notes` في Obsidian/Logseq):**
   - تحويل كل فرع رئيسي إلى صفحة مستقلة داخل `docs/knowledge_map/nodes/` مع الإبقاء على `NourAI_Knowledge_Map.md` كـ **Map of Content (MOC)** مركزي.
3. **استخدام استعلامات ديناميكية (`Dataview / Logseq Queries`):**
   - استخراج لوحة مهام فورية عبر الاستعلام عن `type:: task` و`priority:: P0`، ولوحة مخاطر عبر `type:: gap`.
4. **ربط سجل القرارات المعمارية (`ADR — Architecture Decision Records`):**
   - إنشاء عقدة `[[ADR-001: ByteTokenizer vs BPE]]` و`[[ADR-002: Stdlib HTTP vs FastAPI]]` لتوثيق *لماذا* اتُّخذ كل قرار هندسي، مما يمنع تكرار النقاشات مستقبلاً.
---

## 🚀 7. تحديث التحول إلى NourAI `v3.0.0` (Implemented Autonomous AI Architecture)

تم تنفيذ التحول الفعلي من `v1.2.0` إلى `v3.0.0` مع الحفاظ التام على التوافقية الخلفية (`v1.2.0 -> v1.5 -> v2.0 -> v2.5 -> v3.0`):

- 🤖 **[[Autonomous Agent Core v3.0]]** (`nourai/agent/`): 6 محركات مستقلة (`PerceptionEngine`, `ReasoningEngine`, `PlanningEngine`, `TaskManager`, `CriticEngine`, `RecoveryEngine`) + منسّق `AutonomousAgentOrchestrator` بدورة حياة من 13 مرحلة مع نقاط موافقة بشرية وحماية من الحلقات اللانهائية.
- 🧠 **[[Four-Tier Advanced Memory]]** (`nourai/memory/advanced_memory.py`): ذاكرة عاملة + عرضية + دلالية + إجرائية في `SQLite WAL` مع بحث `TF-IDF` دلالي، إزالة تكرار `SHA-256`، سياسات انتهاء `TTL`، فحص تناقض، وحماية خصوصية تمنع تخزين الأسرار.
- 🕸️ **[[Integrated Knowledge Graph]]** (`nourai/knowledge/knowledge_graph.py`): استخراج علاقات متعددة اللغات (AR/EN/FR)، روابط ثنائية الاتجاه، كشف تناقضات، وتصدير متزامن إلى **Obsidian/Logseq Markdown** و**Freeplane OPML 2.0** و**TheBrain JSON**.
- 🛠️ **[[Standardized Tool Engine]]** (`nourai/tools/`): 9 أدوات معزولة عبر 8 فئات (`file_read`, `file_write` مع Rollback وموافقة صريحة، `document_analysis`, `database_query` قراءة فقط، `web_search`, `report_generation`, `python_sandbox` عبر AST آمن، `api_call` محمي من SSRF، `knowledge_retrieval`) مع سجل تدقيق `SQLite`.
- 🔤 **[[ByteLevelBPETokenizer]]** (`nourai/tokenizer/byte_bpe.py`): مُرمِّز بايت-BPE غير فاقد ببايثون نقي (يدعم `8K`/`16K`) + `L2AnchorRegularizer` لحماية SFT من النسيان الكارثي + تسريع `KV-Cache` مُقاس بمقدار **`2.31x`** على المعالج المركزي (`102.11ms` مقابل `235.49ms`).
- 📊 **[[V3 Evaluation & Regression Gate]]** (`nourai/evaluation/v3_benchmarks.py`): قياس حقيقي (`task_success_rate=1.0`, `tool_selection_accuracy=1.0`, `memory_retrieval_accuracy=1.0`, `failure_recovery_rate=1.0`, `RAM=296.2 MiB`) وبوابة منع تراجع، وواجهات `/api/v3/*` متكاملة في `api/backend.py`.
- 📖 **التوثيق المحدّث (`docs/v3/`):** [ARCHITECTURE_V3.md](file:///d:/NourAI/docs/v3/ARCHITECTURE_V3.md) · [AGENT_ARCHITECTURE.md](file:///d:/NourAI/docs/v3/AGENT_ARCHITECTURE.md) · [NEURAL_MODEL_DESIGN.md](file:///d:/NourAI/docs/v3/NEURAL_MODEL_DESIGN.md) · [MEMORY_ARCHITECTURE.md](file:///d:/NourAI/docs/v3/MEMORY_ARCHITECTURE.md) · [TOOL_SYSTEM.md](file:///d:/NourAI/docs/v3/TOOL_SYSTEM.md) · [SECURITY_MODEL.md](file:///d:/NourAI/docs/v3/SECURITY_MODEL.md) · [TRAINING_GUIDE.md](file:///d:/NourAI/docs/v3/TRAINING_GUIDE.md) · [EVALUATION_REPORT.md](file:///d:/NourAI/docs/v3/EVALUATION_REPORT.md) · [MIGRATION_GUIDE.md](file:///d:/NourAI/docs/v3/MIGRATION_GUIDE.md) · [ROADMAP_V3.md](file:///d:/NourAI/docs/v3/ROADMAP_V3.md).

---

## ⚡ 8. ترقية الذكاء الواقعي والاستقلالية — NourAI `v3.1.0` (`dev/v3.1`)

- 🔬 **[[Neural Forensic Audit & v3.1 Checkpoint]]** (`nourai/neural/v31_training.py`, `checkpoints/v3_1/nourai_v3_1_model.pt`): كشف سبب الحيرة العالية في `releases/v1.2.0/nourai_v1_2_model.pt` (نموذج `Small` منهار بحيرة `EN: 1198.60`)، وتوسيع بيانات SFT إلى `359` تدريب / `71` تحقق / `73` اختبار (`0 Leakage`)، وتدريب نقطة حفظ `v3.1` محمية من النسيان الكارثي بـ `L2AnchorRegularizer` ومزج الإعادة (`Replay Mixing`) لتخفيض الحيرة غير المرئية إلى `AR: 5.27`, `EN: 22.62`, `FR: 16.91`.
- 🧪 **[[105-Task Real-World Agent Benchmark]]** (`nourai/evaluation/v31_agent_benchmark.py`): 105 مهمة واقعية عبر 11 فئة مع تحقق مستقل لكل نتيجة متوقعة (وليس مجرد حالة الإتمام).
- 🧠 **[[Multilingual Normalized Memory & Preference Superseding]]** (`nourai/memory/advanced_memory.py`): توحيد الحروف العربية وإزالة التشكيل وطيّ التشكيل الفرنسي، وتصفية الكلمات الوظيفية مع تعزيز `Semantic/Procedural` وتحديث تفضيلات المستخدم دون تناقض زائف.
- 🔀 **[[DAG Parallel Tool Orchestrator]]** (`nourai/tools/dag_executor.py`): تنفيذ طوبولوجي للخطط مع تشغيل موازٍ آمن لخطوات القراءة المستقلة وتسلسلي للخطوات التابعة أو التعديلية.
- 🩺 **[[Self-Evaluation Engine]]** (`nourai/evaluation/self_eval.py`): تصنيف أسباب الفشل تلقائياً، توليد اختبارات انحدار مرشحة، ومقارنة الإصدارات مع منع التحديثات التلقائية غير المُتحقق منها.
- 🖥️ **[[NourAI Control Center Dashboard]]** (`ui/control_center.html`, `GET /control-center`, `GET /api/v3/control-center/summary`): لوحة تحكم احترافية من 10 وحدات لمراقبة الوكيل والجدول الزمني والذاكرة والموافقات والعتاد والتقييم.
- 📖 **توثيق `v3.1` (`docs/v3.1/`):** [NEURAL_IMPROVEMENTS.md](file:///d:/NourAI/docs/v3.1/NEURAL_IMPROVEMENTS.md) · [AGENT_BENCHMARK.md](file:///d:/NourAI/docs/v3.1/AGENT_BENCHMARK.md) · [MEMORY_VALIDATION.md](file:///d:/NourAI/docs/v3.1/MEMORY_VALIDATION.md) · [DASHBOARD_DESIGN.md](file:///d:/NourAI/docs/v3.1/DASHBOARD_DESIGN.md) · [SECURITY_AUDIT.md](file:///d:/NourAI/docs/v3.1/SECURITY_AUDIT.md) · [FINAL_REPORT.md](file:///d:/NourAI/docs/v3.1/FINAL_REPORT.md).

---

## 🔬 9. الذكاء المستقل المتقدم والبحث والمشاريع طويلة المدى — NourAI `v3.2.0` (`dev/v3.2`)

- 🔍 **[[Autonomous Research Agent]]** (`nourai/research/research_agent.py`): تفكيك الأسئلة البحثية، مقارنة المصادر المستقلة، كشف التناقضات، تقييم موثوقية المصادر، فصل الحقائق الموثقة بالاستشهادات عن الفرضيات، وتحديد الأدلة الناقصة.
- 📋 **[[Persistent Long-Term Project Manager]]** (`nourai/projects/project_manager.py`): إنشاء المشاريع والمعالم والمهام التابعة (`depends_on`)، حفظ الحالة في `SQLite WAL` عبر إعادة التشغيل الكاملة، كشف المهام المحظورة، نقاط الموافقة البشرية، واستئناف المشاريع المتوقفة.
- 🧠 **[[Advanced Reasoning Engine v3.2]]** (`nourai/agent/reasoning.py`): تحليل القيود، توليد الفرضيات القابلة للاختبار، التحقق الرياضي متعدد الخطوات، وكشف تناقضات الذاكرة.
- 📄 **[[Multimodal Document Intelligence]]** (`nourai/documents/multimodal.py`): دعم `TXT/MD/JSON/CSV/DOCX/PDF/Images`، استخراج الجداول (`MD/CSV/JSON/DOCX`)، استخلاص `UTF-16BE` العربي، التقطيع الموثّق بالاستشهادات `[doc:file#chunk_N]`، وكشف صادق لحالة `OCR` دون ادعاء فهم بصري غير متاح.
- 📊 **[[7-Domain Zero-Leakage Neural Datasets & v3.2 Checkpoint]]** (`nourai/neural/v32_datasets.py`, `checkpoints/v3_2/nourai_v3_2_model.pt`) + **[[205-Task Real-World Benchmark]]** (`nourai/evaluation/v32_benchmark.py`).
- 🖥️ **[[Control Center 2.0]]** (`ui/control_center.html`, `/api/v3/research`, `/api/v3/projects`, `/api/v3/documents/multimodal`, `compute_v32_monitoring_snapshot`).
- 📖 **توثيق `v3.2` (`docs/v3.2/`):** [RESEARCH_AGENT.md](file:///d:/NourAI/docs/v3.2/RESEARCH_AGENT.md) · [PROJECT_MANAGER.md](file:///d:/NourAI/docs/v3.2/PROJECT_MANAGER.md) · [REASONING_ENGINE.md](file:///d:/NourAI/docs/v3.2/REASONING_ENGINE.md) · [MULTIMODAL_SYSTEM.md](file:///d:/NourAI/docs/v3.2/MULTIMODAL_SYSTEM.md) · [NEURAL_EVALUATION.md](file:///d:/NourAI/docs/v3.2/NEURAL_EVALUATION.md) · [SECURITY_AUDIT.md](file:///d:/NourAI/docs/v3.2/SECURITY_AUDIT.md) · [FINAL_REPORT.md](file:///d:/NourAI/docs/v3.2/FINAL_REPORT.md).

