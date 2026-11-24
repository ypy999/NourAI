# Jebrex AI — Cloud-First & Storage-Efficient Architecture (`CLOUD_STORAGE_STRATEGY.md`)

- **تاريخ الاعتماد:** `2026-11-24`
- **الهدف:** تقليل استهلاك التخزين المحلي (`C: 6.72 GiB` حرة) وتخفيض حجم مستودع Git (`170.03 MiB` حالياً من ملفات `.pt` الموروثة) والاعتماد على بنية سحابية معيارية منخفضة التكلفة تلتزم بحدود وشروط كل خدمة.

---

## 1. تشخيص التخزين الحالي (الأدلة الفعلية)

| الطبقة | الوضع الحالي المقاس | المشكلة / الفجوة | الإجراء المعتمد |
|---|---:|---|---|
| **ملفات `.pt` المتتبعة في Git** | `19` ملفاً (`170.025 MiB` = `97.3%` من المستودع) | تضخيم Git وتكرار `nourai_v1_2_model.pt` (`18.61 MiB` مرتين) | عدم إضافة أي `.pt` جديد إلى Git عبر `.gitignore` + توثيق جميع الأوزان في `manifests/artifacts_manifest.json` بـ `SHA-256` |
| **الكود والوثائق والاختبارات** | `~4.6 MiB` (`168` ملف `.py` + `159` `.json` + `92` `.md`) | حجم مثالي لـ Git العادي | يبقى في GitHub مع CI/CD |
| **الأرشيفات الخام خارج الـ Worktree (`D:\NourAI\data`)** | `~20.8 GB` (`raw 4.1 GB` + `processed 16.3 GB` + `metadata 0.35 GB`) | أكبر من السعة الحرة لقرص `C:` (`6.72 GiB`) | تبقى خارج Git بالكامل؛ تُرفع لاحقاً إلى GCS/R2/S3 بملف `manifest` ومفاتيح `SHA-256` |

---

## 2. توزيع الطبقات على الخدمات السحابية (Service Separation & Free-Tier Honesty)

| الطبقة | الخدمة الموصى بها | حدود الخطة المجانية الفعلية (موثّقة) | الضمان الصادق للتوفر | سياسة الاستعمال والخصوصية |
|---|---|---|---|---|
| **Code, Versioning, CI/CD** | **GitHub** (`github.com/ypy999/NourAI`) | غير محدود للمستودعات العامة؛ `2,000` دقيقة Actions/شهر للخاص (مفتوحة أكثر للعام) | `99.9%+` | كود، اختبارات، manifests نصية، ووثائق فقط — **صفر أسرار وصفر أوزان جديدة** |
| **Static Web UI & Global CDN** | **Firebase Hosting** (`ypyai-95ca7`, Spark Plan) | `10 GB` تخزين، `360 MB/day` نقل بيانات، HTTPS تلقائي + Preview Channels | **متاح `24/7` فعلياً (بلا Sleep)** ضمن حصة `360 MB/day` | يستضيف `ui/` فقط (`~180 KB` — يكفي لآلاف الزيارات اليومية) |
| **Backend API (ميزانية \$0)** | **Render Free Web Service** (أو **Cloud Run Free Tier**) | Render Free: `512 MB RAM`, `750` ساعة/شهر، **يدخل في Sleep بعد 15 دقيقة خمول**؛ Cloud Run Free: `2M` طلب/شهر + Scale-to-Zero | **ليس `24/7` دائماً** على Render Free (إيقاظ بارد `30–60s`)؛ يُصرّح بذلك للمستخدم | يتطلب ترقية مدفوعة (`\$7/شهر` Render Starter أو `min-instances=1` في Cloud Run) لتشغيل بلا Sleep |
| **Model Weights & Checkpoints** | **Google Cloud Storage** / **Cloudflare R2** (S3-compatible) أو **Google Drive** للتجارب | Cloudflare R2: `10 GB` مجاناً بلا رسوم خروج (Egress `\$0`)؛ GCS Always Free: `5 GB` (`us-west1/central1/east1`)؛ Google Drive: `15 GB` مجاني | تخزين كائنات دائم (`99.999999999%` في GCS/R2) | الأوزان والـ Checkpoints تُفهرس بـ `manifests/artifacts_manifest.json` وتُحمّل عند الطلب مع فحص `SHA-256` |
| **GPU Training Experiments** | **Google Colab** / **Kaggle** (مجاني للتجارب) أو RunPod/Lambda (للتدريب الجاد) | Colab Free: T4 متغير المدة (جلسات تنقطع بعد ساعات)؛ Kaggle: `~30` ساعة GPU/أسبوع | تجارب متقطعة فقط — **إلزامية الاستئناف عبر `cloud_trainer.py`** | مزامنة تلقائية للـ Checkpoint الأخير إلى Drive/GCS/R2 مع حفظ `AdamW` و`Scheduler` |
| **Production Database** | **SQLite WAL** (محلياً) ← **Cloud SQL / Neon / Turso** (عند التوسع السحابي) | Neon/Turso: خطط مجانية بحدود تخزين محددة؛ SQLite محلي غير مستمر على حاويات Render المجانية سريعة الزوال | دائم محلياً؛ **زائل على Render Free** ما لم يُربط بقاعدة خارجية أو قرص دائم | نسخ احتياطية مشفرة مع بصمة `SHA-256` قبل أي ترقية |

---

## 3. القواعد الإلزامية للتخزين وسلامة الملفات

1. **منع تضخم Git:** لا يُسمح برفع مجموعات بيانات ضخمة أو ملفات `.pt` / `.safetensors` / `.gguf` / `.bin` جديدة إلى Git العادي؛ يُستخدم **Git LFS** فقط للملفات المتوسطة التي يحتاجها CI مباشرةً وبقرار موثّق.
2. **التحقق بالبصمات (`Manifest + SHA-256`):** كل وزن نموذج أو تقسيم بيانات يُسجل في `manifests/artifacts_manifest.json` أو `data/manifests/*.manifest.json` وتتحقق أداة `scripts/artifact_storage.py` من تطابق `SHA-256` قبل التحميل.
3. **التحميل الكسول والتخزين المؤقت (`Lazy Download + Local Cache`):** تنزيل الملفات المطلوبة فقط للتشغيل الحالي وحفظها في ذاكرة تخزين مؤقت (`JEBRX_CACHE_DIR` أو `data/cache/`) لتفادي إعادة التحميل.
4. **سياسة الاحتفاظ (`Retention Policy`):** الاحتفاظ دائماً بآخر `best.pt` سليم + آخر `2` نسخ `last.pt` في `history/`؛ **يُحظر حذف آخر نسخة سليمة تلقائياً**.
5. **منع التكرار بين الخدمات:** مصدر حقيقة واحد لكل أصل (Single Source of Truth)؛ عدم تكرار نفس ملف الأوزان في `checkpoints/` و`releases/` في الإصدارات القادمة (الاستعاضة بمؤشر Manifest و`SHA-256`).
6. **تقرير شهري آلي للتخزين والتكاليف:** توليد لقطة آلية عبر `python -m scripts.artifact_storage --report` لمتابعة استهلاك Git، القرص المحلي، والسحابة شهرياً.
