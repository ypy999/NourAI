# Jebrex AI — Phase H Free GPU Platform Discovery & Ranking Report

- **التاريخ:** `2026-11-25`
- **الفرع:** `comprehensive_project_audit`
- **المصدر الموثَّق:** `reports/production_execution/phase_h_gpu_discovery.json` + `training/gpu_discovery.py`
- **التكلفة الفعلية المصروفة:** `\$0.00` (صفر اشتراكات، صفر بطاقات، صفر تكاليف غير مصرح بها)
- **الحالة التنفيذية الصادقة:** `LOCAL_CPU_ONLY_CLOUD_GPU_AWAITING_OWNER_LOGIN`

---

## 1. جرد بيئة التشغيل المحلية الفعلية (Local Hardware & Tooling Audit)

تم فحص جهاز العمل المحلي عبر `training/gpu_discovery.py::probe_local_hardware()` دون افتراض وجود GPU أو تزييف CUDA:

| البند | القيمة المقاسة فعلياً |
|---|---|
| نظام التشغيل | `Windows 10 (AMD64)` |
| المعالج والأنوية المنطقية | `Intel Core i3-10110Y` (`4` خيوط منطقية) |
| الذاكرة العشوائية (RAM) | `7.89 GiB` إجمالي (`2.87 GiB` متاح وقت الفحص) |
| القرص `C:` (مساحة العمل) | `60.04 GiB` إجمالي (`6.20 GiB` متاح فعلياً) |
| Python / PyTorch | `Python 3.15.2` / `torch 2.15.0+cpu` |
| CUDA / GPUs محلياً | **`False`** (`0 GPUs`, `0.0 GiB VRAM`, `FP16=False`, `BF16=False`) |
| PyTorch SDPA | متاح (`sdpa_available=True`)؛ `flash_attention_hardware_capable=False` |
| قياس مصغّر (512×512 FP32 CPU) | `2.91 ms` (`92.26 GFLOPS`) |
| أدوات CLI المثبتة | `git=True`, `firebase=True` — **جميع أدوات السحابة (`kaggle`, `gcloud`, `aws`, `az`, `hf`, `lightning`, `rclone`, `docker`) غير مثبتة (`False`)** |
| مفاتيح/بيانات اعتماد السحابة | جميع متغيرات البيئة و`~/.kaggle/kaggle.json` غائبة (`False`) — صفر أسرار في المستودع |

---

## 2. مقارنة منصات GPU المجانية والمفتوحة (تحقق فعلي لعامي 2025/2026)

لم نفترض بقاء أي حصة مجانية قديمة؛ بل جرى التحقق الميداني من الحصص والشروط الحالية لـ **10 منصات** وتصنيفها بواسطة محرك الترتيب الفعلي (`rank_platforms_for_model`) على نموذج `jebrex-125m` (`124.275M` معلمة، هدف تجريبي `100M` توكن) وفق **الأداء التدريبي القابل للتسليم أسبوعياً وموثوقية حفظ Checkpoints والوصول الذاتي** (وليس اسم الـ GPU وحده):

| الترتيب | المنصة | نوع GPU وVRAM | الساعات المجانية / الجلسة | BF16 / FP16 | التكلفة | التقييم الفعلي (من 100) | حالة الأهلية | الدور الموصى به |
|---:|---|---|---|---|---|---:|---|---|
| **1** | **Kaggle Notebooks Free** | `NVIDIA Tesla T4 × 2` (`32 GB` إجمالي، `16 GB/GPU`) | `30 ساعة/أسبوع` متجددة / `12 ساعة` للجلسة | FP16 ✅ / BF16 ❌ | `\$0.00` | **`91.03`** | `ELIGIBLE_OWNER_AUTH_REQUIRED` | **المنصة الأساسية لتدريب `125M`** (تشغيل خلفي `Save & Run All` + `~1.39B` توكن/أسبوع) |
| **2** | **Lightning AI Studios Free** | `NVIDIA L4 (24 GB)` أو `T4 (16 GB)` | `30–50 رصيد ترويجي` مرة واحدة (`~22–50 ساعة`) | BF16 ✅ / FP16 ✅ | `\$0.00` | **`85.54`** | `ELIGIBLE_OWNER_AUTH_REQUIRED` | **دفعة تسريع ثانية بـ BF16** وتدريب SFT/تقييم سريع (غير متجدد شهرياً) |
| **3** | **Google TPU Research Cloud (TRC)** | `Cloud TPU v5e/v6e` (`128 GB+` للمنحة) | منحة بحثية مفتوحة (`168 ساعة/أسبوع`) | BF16 ✅ / XLA | `\$0.00` | **`83.00`** | `ELIGIBLE_APPLICATION_REQUIRED` | **منحة بحثية طويلة المدى** لتوسيع `125M (2.5B)` و`350M (7B)` بعد الموافقة اليدوية |
| **4** | **Google Colab Free** | `NVIDIA T4 (16 GB)` (أو L4 نادراً) | حصة ديناميكية غير معلنة (`~15 ساعة/أسبوع` تقديراً) | FP16 ✅ / BF16 ❌ | `\$0.00` | **`80.20`** | `ELIGIBLE_OWNER_AUTH_REQUIRED` | **منصة مساعدة** للاستئناف التفاعلي والتجارب عبر ربط Google Drive |
| **5** | **Paperspace Gradient Free** | `NVIDIA Quadro M4000 (8 GB)` | `6 ساعات/جلسة` (توفر محدود جداً) | بدون Tensor Cores | `\$0.00` | هامشي | `MARGINAL_LOW_VRAM` | محدود جداً (`8 GB VRAM` + `5 GB` تخزين)؛ يصلح لـ `micro` فقط |
| **—** | **Hugging Face ZeroGPU Free** | `NVIDIA H200` (مشاركة ديناميكية) | **`~3.5–5 دقائق GPU/يوم`** (`60–300s` للنداء) | BF16 ✅ | `\$0.00` | **`0.00`** | **`DISQUALIFIED_NOT_FOR_PRETRAINING`** | للاستدلال وعروض Gradio فقط (`@spaces.GPU`) — **مستبعد للتدريب المسبق** |
| **—** | **AWS SageMaker Studio Lab / Free Tier** | مغلق للتسجيل الجديد؛ Free Tier = CPU فقط | `0 ساعة GPU` للحسابات الجديدة | ❌ | `\$0.00` | **`0.00`** | **`DISQUALIFIED_CLOSED_OR_NO_GPU`** | غير متاح للحسابات الجديدة |
| **—** | **Google Cloud Free Trial (`\$300`)** | **حصة GPU = `0`** | محظور رفع حصة GPU دون ترقية لحساب مدفوع | ❌ | `\$0.00` | **`0.00`** | **`BLOCKED_ZERO_GPU_QUOTA`** | مستبعد للتدريب بـ `\$0`؛ مفيد لـ `5 GB` GCS Always-Free فقط |
| **—** | **Azure for Students (`\$100`)** | **حصة N-series GPU = `0`** | محظور رفع حصة GPU دون Pay-As-You-Go | ❌ | `\$0.00` | **`0.00`** | **`BLOCKED_ZERO_GPU_QUOTA`** | غير صالح لتدريب GPU مجاناً |
| **—** | **Oracle Cloud Always-Free** | **بدون GPU** (`4 ARM Ampere OCPU + 24 GB RAM`) | دائم `24/7` بدون Sleep | ❌ | `\$0.00` | **`0.00`** | **`DISQUALIFIED_NO_GPU`** | **الأفضل لاستضافة خادم API بـ Docker 24/7 مجاناً** (وليس تدريب GPU) |

---

## 3. تقديرات الذاكرة والزمن قبل التشغيل (Pre-Flight Memory & Time Estimator)

قبل حجز أو استهلاك دقيقة واحدة من حصة `Kaggle T4 × 2` المجانية، حسب المحرك (`estimate_memory_and_time`) متطلبات الذاكرة والوقت لكل مرحلة (مع `gradient_checkpointing=True`، `seq_len=1024`، `FP16` + `AdamW FP32`، ودُفعة فعالة `65,536` توكن/خطوة):

| المرحلة / النموذج | المعلمات الدقيقة | ذروة VRAM / GPU | يلائم `16 GB T4`؟ | الدفعة الموصى بها (`mb × accum × gpu`) | السرعة المقدّرة | الساعات المطلوبة | عدد الجلسات (`12h`) | الأسابيع المجانية (`30h/wk`) |
|---|---:|---:|:---:|---|---:|---:|---:|---:|
| `jebrex-micro` (`1M` pilot) | `1,083,520` | `0.77 GB` | نعم ✅ | `16 × 8 × 2` (`seq=256`) | `~154,440 tok/s` | `< 0.05 h` | `1` | `0.00` |
| `jebrex-125m` (`100M` pilot) | `124,275,456` | `5.405 GB` | نعم ✅ | `16 × 2 × 2` (`seq=1024`) | `~12,870 tok/s` | **`2.16 h`** | **`1`** | **`0.07`** |
| `jebrex-125m` (`2.5B` Chinchilla) | `124,275,456` | `5.405 GB` | نعم ✅ | `16 × 2 × 2` (`seq=1024`) | `~12,870 tok/s` | **`53.96 h`** | **`6`** | **`1.80`** |
| `jebrex-350m` (`7.0B` Chinchilla) | `351,323,136` | `9.898 GB` | نعم ✅ | `16 × 2 × 2` (`seq=1024`) | `~4,553 tok/s` | **`427.11 h`** | **`42`** | **`14.24`** |

---

## 4. خطوات تسجيل الدخول والتحقق المطلوبة من مالك المشروع (صفر تجاوز للصلاحيات)

التزاماً بالقاعدة الصارمة في البند 4 (عدم تجاوز التحقق أو إنشاء حسابات متعددة أو حفظ أسرار في المستودع)، توقف التنفيذ السحابي بانتظار الإجراءات التالية من المالك:

1. **Kaggle Notebooks (أساسي — `\$0.00`):** تسجيل الدخول بحساب المالك وإكمال تحقق الهاتف SMS وقبول الشروط لتفعيل `T4 × 2` والإنترنت، ورفع `notebooks/JebrexAI_Cloud_GPU_Training.ipynb` (أو وضع `~/.kaggle/kaggle.json` محلياً خارج Git).
2. **Cloudflare R2 Free (`10 GB`، صفر رسوم Egress) أو Google Drive (`15 GB`):** إنشاء حاوية خارجية لتخزين Checkpoints وShards مع حقن المفاتيح في `Kaggle Secrets` / `Colab Secrets` فقط.
3. **Lightning AI Studios (اختياري لدفعة BF16 — `\$0.00`):** تسجيل الدخول وتحقق الهاتف لتفعيل الأرصدة الترويجية المجانية دون بطاقة.
4. **Google Colab Free (اختياري للاستئناف التفاعلي — `\$0.00`):** فتح الدفتر بحساب Google الخاص بالمالك.
5. **Oracle Cloud Always-Free (اختياري لاستضافة API `24/7` — `\$0.00`):** إنشاء مثيل `4 ARM OCPU + 24 GB RAM` لاستضافة خادم الاستدلال فقط.
