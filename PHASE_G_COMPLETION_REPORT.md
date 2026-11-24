# Jebrex AI — Phase G Completion & Decision Report (`PHASE_G_COMPLETION_REPORT.md`)

**Phase:** `PHASE G — Multilingual Production Dataset & Cloud GPU Training`  
**Audit & Verification Date (UTC):** `2026-11-24`  
**Repository & Branch:** `https://github.com/ypy999/NourAI/tree/comprehensive_project_audit`  
**Live Preview Website (24/7 Firebase Hosting CDN):** **`https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app`** (Release `e28c0aefa50748ff`, `HTTP 200`, `321.3 ms`, يعرض `model_version_id` الحقيقي)  
**Primary Live Channel (`https://ypyai-95ca7.web.app`):** محفوظ كما هو دون استبدال حتى اجتياز نموذج جديد لاختبارات القبول وموافقة المستخدم.

> **إفصاح الصدق العلمي الصريح (Mandatory Scientific Integrity Declaration):**
> - **حالة اكتمال المرحلة G:** **لم يُعلن اكتمال التدريب الإنتاجي بعد (`ENGINEERING_COMPLETE__PRODUCTION_TRAINING_BLOCKED_PENDING_APPROVAL`)**.
> - جميع خطوط البيانات المرخّصة، سجل التراخيص، الشحن السحابي بالدفعات مع SHA-256، مقارنة التخزين والـ GPU، حارس استقرار التدريب، التقييم المرجعي متعدد الإصدارات، وبوابة الترقية **مبنية ومختبرة آلياً**؛ بينما **لم يتم تشغيل تدريب `125M` أو `350M` الفعلي بعد** التزاماً بقواعد: **(1) عدم تجاوز `DATA_NOT_READY`** و**(2) عدم تفعيل أي خدمة سحابية مدفوعة قبل تقديم مقارنة واضحة والحصول على موافقة المستخدم**.

---

## 1. الاستحواذ على البيانات وسجل التراخيص (`Data Acquisition & License Registry`)

### أ) سجل المصادر والتراخيص الموثّق (`data/phase_g_license_registry.json`)
تفرض البوابة البرمجية `LicensePolicyGate` (`data/phase_g_acquisition.py`) استبعاداً صارماً لأي مصدر مجهول الترخيص أو غير تجاري (`LicenseViolationError`):

| المصدر (`dataset_id`) | اللغات | الترخيص (`SPDX`) | استخدام تجاري | إعادة توزيع | الحكم المعتمد (`audit_status`) |
|---|---|:---:|:---:|:---:|---|
| **`tatoeba_cc0`** | `ar, en, fr` | `CC0-1.0` | **نعم** | **نعم** | `APPROVED_COMMERCIAL_AND_REDISTRIBUTION` (مُستحوَّذ ومُختبر) |
| **`pleias_common_corpus`** | `ar, en, fr` | `Public Domain / Open` | **نعم** | **نعم** | `APPROVED_COMMERCIAL_AND_REDISTRIBUTION` (للرفع السحابي على R2) |
| **`fineweb_2_multilingual`** | `ar, en, fr` | `ODC-By-1.0` | **نعم** | **نعم** (بإسناد) | `APPROVED_COMMERCIAL_WITH_ATTRIBUTION` (العمود الأساسي لـ `3B–10B` توكن) |
| **`europarl_en_fr`** | `en, fr` | `Public Domain EU` | **نعم** | **نعم** | `APPROVED_COMMERCIAL_WITH_ATTRIBUTION` |
| **`qnl_arabic_ocr_v2`** | `ar` | `Public Domain + CC0 Meta` | **نعم** (تدريب محلي) | **لا** | `APPROVED_TRAINING_NO_ARCHIVE_REDISTRIBUTION` (يتطلب فلتر OCR صارم) |
| **`wto_public_corpus_en_fr`** | `en, fr` | `WTO Public Terms` | **نعم** (بإسناد) | **لا** | `APPROVED_TRAINING_NO_ARCHIVE_REDISTRIBUTION` |
| **`wikimedia_wikipedia_ar_en_fr`** | `ar, en, fr` | `CC-BY-SA-4.0` | **نعم** | **مقيّد بـ SA** | `NEEDS_SHARE_ALIKE_LEGAL_REVIEW` (موقوف حتى مراجعة شرط ShareAlike) |
| `oscar_web` / `culturax_web` | `ar, en, fr` | `UNKNOWN` | **لا** | **لا** | **`REJECTED_UNKNOWN_LICENSE`** (مرفوض برمجياً) |
| `madar_parallel` | `ar` | `Research Non-Commercial` | **لا** | **لا** | **`REJECTED_NONCOMMERCIAL`** (مرفوض برمجياً) |

### ب) نتائج الاستحواذ الفعلي على حزمة الاختبار المرخّصة (`Tatoeba CC0-1.0`)
تم تنزيل الأرشيف الرسمي والتحقق من `SHA-256` وتخزينه خارج GitHub في `.gitignore`d `data/phase_g_pilot/` مع إزالة التكرار وفصل التقسيمات وقياس توكنات BPE الفعلية (الدليل: `reports/production_execution/phase_g_data_acquisition.json` و`data/manifests/phase_g_tatoeba_cc0_pilot.v1.0.0.manifest.json`):
- **الأرشيف:** `sentences_CC0.tar.bz2` (`7.59 MB`, SHA-256: `ffece08afdd32f25cf1ae12040a41991e757df322eef93fbde40d51f0cd205bb`).
- **الجمل الفريدة المستخرجة:** **`39,185`** جملة (حُذفت `19` نسخة مكررة، **صفر `0` تسرب بين `train` و`validation` و`test`**).
- **إجمالي توكنات BPE المقاسة فعلياً (`jebrex_bpe_v1.json`):** **`1,122,564` توكن BPE**:
  - الإنجليزية (`en`): **`754,885`** توكن BPE
  - الفرنسية (`fr`): **`367,642`** توكن BPE
  - **العربية (`ar`): `37` توكن BPE فقط!** *(اكتشاف تجريبي مهم يثبت أن حزمة Tatoeba CC0 فقيرة جداً بالعربية، وأن بلوغ `1B` توكن عربي يعتمد أساساً على سحب **`FineWeb-2 Arabic` (`ODC-By-1.0`)** و**`PleIAs Common Corpus`** و**`QNL Arabic OCR v2`** المفلتر على تخزين R2 السحابي).*
- **بوابة الإنتاج:** تبقى **`DATA_NOT_READY`** (`1.12M` مقابل `3,000,000,000` توكن مطلوب للمستوى 1).

---

## 2. مقارنة التخزين السحابي (`Cloud Storage Comparison`)

مصدر الحسابات: بحث حي (`2026-11-24`) + `scripts/phase_g_cost_model.py` (`reports/production_execution/phase_g_cost_report.json`):

| الخدمة | الحصة المجانية | سعر التخزين المدفوع | رسوم نقل البيانات (`Egress`) | تكلفة 25 GB (المرحلة 1 + 10 سحبات GPU) | تكلفة 75 GB (المرحلة 2 + 10 سحبات GPU) | التقييم الهندسي |
|---|---|---|---|---:|---:|---|
| **Cloudflare R2 (الموصى به)** | **`10 GB/mo`** + 1M Write / 10M Read | `\$0.015 / GB-mo` | **`\$0.00` (مجاني تماماً)** | **`\$0.22 / mo`** | **`\$0.98 / mo`** | **الأفضل للتدريب السحابي** (S3-compatible، صفر رسوم نقل عند تكرار السحب من Colab/Kaggle/Vast/RunPod) |
| **Google Cloud Storage (GCS)** | `5 GB/mo` (مناطق US فقط) | `\$0.020–\$0.026 / GB-mo` | **`\$0.12 / GiB`** (خارج GCP) | **`\$30.40 / mo`** (`\$30` نقل!) | **`\$91.40 / mo`** (`\$90` نقل!) | مكلف جداً خارج GCP؛ يُستخدم فقط إذا كان التدريب داخل نفس منطقة Vertex/GCE |
| **Google Drive** | `15 GB` (مشتركة مع Gmail/Photos) | `\$1.99 / mo` (`100 GB`) | `\$0.00` | `\$1.99 / mo` | `\$1.99 / mo` | مفيد لحفظ Checkpoints من Colab، لكنه مقيّد بحدود API (`429`) وغير ملائم لـ Shards ضخمة |

- **آلية الشحن المطبّقة والمختبرة (`scripts/cloud_shard_sync.py`):**
  - تقسيم البيانات إلى Shards مع `SHA-256` لكل Shard.
  - تنزيل على دفعات محدودة (`BatchedCloudShardFetcher`) مع إخلاء الكاش بعد كل دفعة لمنع امتلاء القرص المحلي/السحابي ورفض أي ملف تالف (`SHA-256 mismatch`).
  - سياسة احتفاظ (`enforce_healthy_checkpoint_retention`) تحفظ آخر نقطة حفظ سليمة دائماً.

---

## 3. مقارنة منصات GPU السحابية (`Cloud GPU Comparison`)

تقديرات الحوسبة وفق معادلة Chinchilla ($C \approx 6 N D$) مع معامل هامش أمان `1.5x–1.8x` للاستئناف والتقييم:
- **المرحلة 1 (`jebrex-125m`: `124.28M` × `3B` توكن):** $\approx 2.24 \times 10^{18}$ FLOPs.
- **المرحلة 2 (`jebrex-350m`: `351.32M` × `10B` توكن):** $\approx 2.11 \times 10^{19}$ FLOPs.

| المنصة / البطاقة | VRAM | الحصة / طبيعة التشغيل | المرحلة 1 (`125M` / 3B توكن) — ساعات / تكلفة | المرحلة 2 (`350M` / 10B توكن) — ساعات / تكلفة | يحتاج موافقة مالية؟ |
|---|---:|---|---:|---:|:---:|
| **Kaggle Notebooks (Free)** | **`32 GB` (`T4 × 2`)** | ~30 ساعة/أسبوع مجاناً (12h/جلسة) | `~32h` / **`\$0.00`** (على أسبوعين) | غير عملي لـ 10B (يتجاوز ~300h) | **لا (`\$0`)** |
| **Google Colab Free** | `16 GB` (`T4`) | حصة متغيرة غير مضمونة (انقطاعات متكررة) | غير مضمون للإكمال الكامل / `\$0.00` | غير مناسب | **لا (`\$0`)** |
| **Google Colab Pro** | `24–80 GB` (`L4/A100`) | 100 CU/شهر + بونص 15–30h/أسبوع Kaggle | `~9h` / **`\$9.99`** | يتطلب Pro+ أو CUs إضافية | **نعم** |
| **Vast.ai (`RTX 4090`)** | `24 GB` | دفع بالثانية (قابل للانقطاع + استئناف SHA-256) | `~9.9h` @ `\$0.38/h` = **`~\$3.76`** | بطيء لـ 350M (يُفضل A100) | **نعم** |
| **Vast.ai (`A100 80GB`)** | `80 GB` | دفع بالثانية (سوق سحابي مرن) | `~7.5h` @ `\$0.85/h` = **`~\$6.38`** | `~70.3h` @ `\$0.85/h` = **`~\$59.75`** | **نعم** |
| **RunPod Community (`A100 80GB`)** | `80 GB` | دفع بالثانية (استقرار أعلى) | `~7.5h` @ `\$1.39/h` = **`~\$10.42`** | `~70.3h` @ `\$1.39/h` = **`~\$97.72`** | **نعم** |
| **RunPod Community (`H100 80GB`)** | `80 GB` | دفع بالثانية (أسرع تدريب) | `~3.2h` @ `\$2.69/h` = **`~\$8.60`** | `~29.3h` @ `\$2.69/h` = **`~\$78.82`** | **نعم** |
| **Lambda Labs (`A100` / `H100`)** | `80 GB` | عناقيد مخصصة، `\$0` Egress | `A100`: `~\$21.37` | `H100`: `~\$103.85` | **نعم** |

---

## 4. خط التدريب التدريجي (`125M` ← `350M`) وحارس الاستقرار

- **عدم تجاوز `DATA_NOT_READY`:** `training/phase_g_orchestrator.py::check_stage_progression_gate("stage_1_125m")` و`"stage_2_350m"` يرفضان تشغيل التدريب الإنتاجي تلقائياً (`PRODUCTION_TRAINING_BLOCKED`) قبل جاهزية البيانات والـ GPU.
- **الانتقال المشروط إلى `350M`:** تمنع الدالة بدء المرحلة 2 (`350M`) قبل وجود ملف تقييم مستقل ناجح للمرحلة 1 (`125M`) يجتاز بوابة القبول (`promotion_passed == True`).
- **حارس استقرار التدريب (`TrainingStabilityWatchdog`):** يراقب كل خطوة (`NaN`/`Inf`، قفزات الخسارة `> 2.5x EMA`، وانفجار `grad_norm`)، ويعيد تحميل آخر Checkpoint سليم تلقائياً مع خفض `LR × 0.5` (مُختبَر آلياً في `test_stage_progression_watchdog_rollback_and_test_isolation`).
- **عزل شريحة الاختبار النهائية (`assert_test_split_not_used_in_training`):** يمنع برمجياً تمرير أي ملف `test` إلى التدريب أو التحقق.

---

## 5. التقييم المرجعي متعدد الإصدارات وبوابة الترقية (`Evaluation & Promotion Gate`)

تم تشغيل التقييم الموحّد لخمسة محاور (`evaluation/phase_g_evaluator.py`) عبر **العربية والإنجليزية والفرنسية** (دون الاعتماد على Perplexity وحده) وحفظ النتائج للمقارنة في `reports/evaluation/versions/` و`reports/evaluation/VERSION_COMPARISON_MATRIX.md`:

| الإصدار (`version_tag`) | المعمارية | المعاملات | Held-Out PPL | التعميم AR/EN/FR | المحادثة متعددة الأدوار | عدم التدهور (Non-Degen) | الأمان AR/EN/FR | السرعة (`tok/s`) | قرار بوابة الترقية |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **`nourai-v1.2.0-small-4.87m`** | `nourai-v1-gpt` | `4.87M` | `417.97` | `0.0%` | `0.0%` | `83.3%` | **`100.0%`** | `37.03` | **`PROMOTION_BLOCKED`** |
| **`jebrex-v2.0-micro-1.08m-sanity`** | `jebrex-v2-modern` | `1.08M` | `1134.58` | `0.0%` | `0.0%` | `0.0%` | **`100.0%`** | `153.61` | **`PROMOTION_BLOCKED`** |

*(أثناء التقييم رصدنا أن عبارة حقن موجه عربية لم تكن مغطاة في `PROMPT_INJECTION_PATTERNS`، فأغلقنا الفجوة فوراً في `nourai/capabilities.py` ليرتفع الأمان عبر اللغات الثلاث إلى **`100.0%`**).*

---

## 6. النشر الحي وعرض الإصدار الحقيقي (`Live Deployment`)

- الموقع التجريبي متاح ومختبر حياً على **`https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app`** (`HTTP 200`, `321.3 ms`, إصدار `e28c0aefa50748ff`).
- أُضيف معرّف الإصدار الحقيقي `model_version_id` (مثال: `nourai-v1-nourai_v1_2_model-4.87m-695da3c3`) إلى `inference/engine.py` و`GET /api/v1/model/status` وعُرض مباشرةً في شريط الحالة الحي بـ `ui/index.html`.
- لم يُستبدل النموذج العامل لأن بوابة الترقية `evaluate_model_promotion_gate()` تمنع الترقية قبل اجتياز جميع معايير القبول.

---

## 7. تقرير التحكم بالتكلفة (`Cost Control Report`)

| البند | الاستهلاك / الحجم الفعلي اليوم | التكلفة الفعلية اليوم | التقدير للمرحلة 1 (`125M` / 3B) | التقدير للمرحلة 2 (`350M` / 10B) |
|---|---|---:|---:|---:|
| **حجم البيانات** | `1.18 MiB` (`jebrex_v1_clean`) + `8.5 MB` (`tatoeba_cc0` pilot، خارج Git) | `\$0.00` | `~18–25 GB` (على R2) | `~65–75 GB` (على R2) |
| **مساحة التخزين السحابي** | `0 GB` مدفوع (محلياً `6.67 GiB` فارغة على `C:`) | `\$0.00` | **R2:** `\$0.22 / mo` | **R2:** `\$0.98 / mo` |
| **تكلفة نقل البيانات (`Egress`)** | `\$0.00` | `\$0.00` | **R2:** `\$0.00` (مقابل `\$30` في GCS!) | **R2:** `\$0.00` (مقابل `\$90` في GCS!) |
| **تكلفة GPU السحابي** | لم تُفعّل أي خدمة مدفوعة | `\$0.00` | `\$0` (Kaggle) أو `\$3.76–\$10.42` (Vast/RunPod) | `\$59.75–\$97.72` (Vast/RunPod A100/H100) |
| **تكلفة الاستضافة** | Firebase Spark (`< 0.25 MiB` من `10 GB`) | `\$0.00` | `\$0.00` (Firebase + Oracle Free/Docker) أو `\$5–\$25/mo` | `\$0.00` أو `\$5–\$25/mo` |
| **التكلفة الإجمالية** | — | **`\$0.00`** | **`\$0` (مجاني) أو `~\$10–\$20` (سريع)** | **`~\$75–\$125` (مع المرحلة 1 وR2)** |

---

## 8. الاختبارات المتحقَّق منها وخيارات القرار بانتظار موافقتك

- **الاختبارات الآلية (`14` ملف اختبار تشمل `test_jebrex_phase_g.py`):** **`61/61` اختباراً ناجحاً (`100%`، `0` إخفاقات، `0` تراجعات)**.
- **التدقيق الأمني (`security/security_audit_v2.py`):** `PASS_NO_OPEN_CRITICAL_OR_HIGH` (`0 CRITICAL`, `0 HIGH`, `3 MEDIUM`, `1 LOW`).

### اختر المسار الذي توافق عليه للانتقال إلى التدريب الفعلي على GPU:
1. **الخيار A (`\$0.00` مجاني بالكامل):** إنشاء Bucket مجاني على **Cloudflare R2 (`10 GB`)** + سحب شريحة متوازنة (`AR/EN/FR`) من `FineWeb-2` و`Common Corpus` + تشغيل تدريب `jebrex-125m` على **Kaggle Notebooks المجانية (`T4×2`، 30 ساعة/أسبوع)** عبر حزم SHA-256 المستأنفة.
2. **الخيار B (`\$10 – \$25` ميزانية منخفضة — المرحلة 1 كاملة `125M` على `3B` توكن):** تخزين **Cloudflare R2 (`\$0.22/mo`)** + استئجار **Vast.ai RTX 4090 / A100** أو **RunPod A100** (`~7.5–10` ساعات) لإتمام تدريب `jebrex-125m` في جلسة واحدة متصلة مع تقييم مستقل كامل.
3. **الخيار C (`\$75 – \$150` — المرحلتان `125M` ← `350M`):** إتمام `125M` أولاً، وعند اجتيازه بوابة القبول الانتقال فوراً إلى `jebrex-350m` (`10B` توكن) على `Vast.ai/RunPod A100/H100` + `Cloudflare R2`.
