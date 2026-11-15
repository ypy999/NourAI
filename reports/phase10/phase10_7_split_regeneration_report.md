# تقرير المرحلة 10.7 — اعتماد سياسة التقسيم وإعادة توليد البيانات (Dataset Split Policy & Regeneration)

## 1. الملخص التنفيذي
تم استكمال **المرحلة 10.7** بنجاح عبر تطبيق سياسة التقسيم المعتمدة رسمياً (`0.90 / 0.05 / 0.05`، بذرة العشوائية `random_seed = 1337`، والتجميع على مستوى الوثيقة `group_field = "document_id"` لمنع التسرب) وإعادة توليد التقسيمات القياسية القابلة لإعادة الإنتاج بالكامل داخل [data/processed/phase10_3/splits/](file:///D:/NourAI/data/processed/phase10_3/splits) مع إصدار بيان التشغيل الذري الموثق [run_manifest.json](file:///D:/NourAI/data/processed/phase10_3/run_manifest.json) وبيان الاعتماد [multilingual_phase10_3.manifest.json](file:///D:/NourAI/data/approved/multilingual_phase10_3.manifest.json).

---

## 2. نتائج التقسيم والتحقق من عدم التسرب

| المؤشر | القيمة المسجلة | حالة القبول |
| --- | --- | --- |
| نسب التقسيم المعتمدة (`train / validation / test`) | `0.90 / 0.05 / 0.05` | مطابق (`0` مخالفات في `check_split_decisions`) |
| النسب الفعلية المحققة | `0.90001 / 0.05000 / 0.05000` | مطابق |
| بذرة العشوائية (`random_seed`) | `1337` | مطابق |
| حقل التجميع لمنع التسرب (`grouping_field`) | `document_id` | مطابق |
| إجمالي السجلات المستوردة | `114,243` (`55,719` Tatoeba + `58` QNL + `58,466` WTO) | مكتمل |
| التكرارات المستبعدة عالمياً قبل التقسيم | `10,573` | مكتمل |
| السجلات الفريدة المقسمة (`unique_records`) | `103,670` (`train: 93,304`, `validation: 5,183`, `test: 5,183`) | مكتمل |
| تسرب النصوص المكررة بين التقسيمات (`duplicate_leakage`) | `0` | اجتياز كامل (`PASS`) |
| تسرب الوثائق بين التقسيمات (`document_group_leakage`) | `0` | اجتياز كامل (`PASS`) |

---

## 3. نتائج تدقيق قابلية إعادة الإنتاج (`audit_artifacts`)

عند تشغيل `audit_artifacts(root, targets=["data/processed/phase10_3"])`:
- إجمالي الملفات المدققة: `16` ملفاً (`9` ملفات بيانات + `7` ملفات sidecars/manifest).
- جميع ملفات البيانات التسعة (`3` ملفات `imported` + `3` ملفات `normalized` + `3` ملفات `split`) حصلت على حكم **`reproducible: "YES"`** (`0` ملفات `NO`، و`0` ملفات `PARTIAL`).
- اجتاز [run_manifest.json](file:///D:/NourAI/data/processed/phase10_3/run_manifest.json) فحص `verify_run_manifest()` بالكامل (`valid: True, errors: []`).
