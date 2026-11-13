# NourAI — تقرير المرحلة الرابعة: الإعدادات ومسار البيانات

## الحالة والبيئة

أُنجزت المرحلة الرابعة على `D:\NourAI`. أُنشئت النسخة الاحتياطية أولًا في `.backup/phase4-before-fix/` للملفات المتأثرة/المراجعة: `configs/default.json`, `model/transformer.py`, `model/__init__.py`, `tokenizer/byte_tokenizer.py`, `tests/test_phase3.py`, و`README.md`. لم تتغير ملفات المرحلة الثالثة؛ ظل Transformer وTokenizer وواجهاتهما كما هي.

استُخدمت البيئة الموجودة فقط: Python 3.15.2، PyTorch 2.15.0+cpu، pytest 9.2.0، NumPy 2.5.4. لم تثبت مكتبات جديدة، ولم يُشغّل التدريب أو CUDA.

## الملفات التي فُحصت

- بنية المشروع ومحتوياته خارج تفاصيل `.venv` والنسخ الاحتياطية.
- `configs/default.json`
- `model/transformer.py`
- `tokenizer/byte_tokenizer.py`
- `tests/test_phase3.py`
- `PHASE_3_REPAIR_REPORT.md`
- بيئة `.venv` وإصدارات حزمها.

لم تُفترض وحدات أو وظائف لم تكن موجودة؛ كانت إعدادات JSON غير مربوطة بإنشاء النموذج، ولم يكن هناك Data Pipeline أو Dataset قبل هذه المرحلة.

## الملفات المنشأة

- `configs/configuration.py`: نموذج إعدادات واحد، تحميل JSON، قيم افتراضية، وفحوص للأنواع والقيم والتوافق.
- `configs/__init__.py`: واجهة استيراد عامة للإعدادات.
- `model/factory.py`: Model Factory تنشئ `GPTConfig` و`GPT` مباشرة من `ProjectConfig` أو JSON وتعيد النموذج مع الإعدادات المحمّلة.
- `data/__init__.py`, `data/processor.py`: قراءة TXT وJSONL، التنظيف، إحصاءات وأخطاء، إزالة تكرار على SQLite، تقسيم ثابت إلى train/validation، وكتابة JSONL للنتائج.
- `datasets/__init__.py`, `datasets/dataset.py`, `datasets/loader.py`: Dataset لتسلسلات next-token بطول ثابت وDataLoader بدفعات الإعدادات.
- `tests/test_phase4.py`: اختبارات المرحلة الرابعة.
- `PHASE_4_AUDIT_REPORT.md`: هذا التقرير.

## الملفات المعدّلة

- `configs/default.json`: أبقى القيم السابقة وأضاف `validation_fraction`, `random_seed`, `max_document_chars`, `io_batch_lines`.

لم يُعدّل Transformer أو Tokenizer أو اختبارات المرحلة الثالثة أو README. Factory يربط الإعدادات بالنموذج دون تغيير الواجهة السابقة.

## نظام الإعدادات وModel Factory

`load_config()` يقرأ UTF-8 JSON، يرفض الجذر غير الكائني والمفاتيح الزائدة والقيم ذات الأنواع/المديات غير الصحيحة. التحقق يشمل توافق `vocab_size` مع `ByteTokenizer`، قسمة `d_model` على `n_heads`، الأبعاد والطبقات الموجبة، dropout، حجم الدفعة، التعلم، نسبة التحقق، seed، وحجوم المعالجة. الإعدادات المستقبلية للتدريب موجودة في الملف نفسه.

`create_model()` يمرر إعدادات البنية إلى `GPTConfig`؛ اختبارات غيّرت `d_model=24`, `n_heads=3`, `n_layers=2`, و`context_length=16` وأثبتت انعكاسها على طبقات النموذج وموضع embeddings. تغيير `configs/default.json` أو ملف JSON بديل يغير النموذج المنشأ فعليًا.

## Data Pipeline وDataset/DataLoader

- يقبل المصدر ملفات TXT وJSONL أو مجلدات تحتويهما. TXT يعامل كل سطر كوثيقة؛ JSONL يتطلب كائنًا بحقل نصي `text` أو `content`.
- يقرأ UTF-8 بصيغة صارمة، ويسجل السطور التالفة/غير الصالحة، ويستمر لبقية السجلات. السجلات الفارغة والتكرارات والمتجاوزة للحد تظهر في `DataPipelineStats`.
- التنظيف يستخدم NFC، يزيل المحارف التحكمية غير المرغوبة، يضغط المسافات ويحافظ على فواصل الفقرات. لا يعدّل ملفات المصدر.
- SQLite على القرص يزيل التطابقات الدقيقة قبل التقسيم. التقسيم حسب hash حتمي للنص وseed، فيبقى النص المكرر في مجموعة واحدة. ملفات النتائج الموجودة لا تُستبدل.
- الإدخال سطري مع حد أقصى لحجم السطر، وقائمة أخطاء التشخيص محدودة لتظل الذاكرة مضبوطة. إزالة التكرار لا تخزن كل النصوص في ذاكرة Python.
- Dataset يقرأ JSONL على دفعات منطقية ويكتب token cache مؤقتًا على القرص، ثم يجلب نافذة token واحدة لكل مثال. ينتج `input_ids` و`target_ids` من إزاحة واحدة، كلاهما `torch.int64` وبطول `context_length`. DataLoader يستخدم `batch_size` من الإعدادات.

## نتائج الاختبارات الفعلية

اختبارات الجزء الأول (الإعدادات وFactory): **9 passed، 6 deselected**.

اختبارات خط البيانات وDataset/DataLoader: **6 passed، 9 deselected**. ظهر أولًا خطأ حقيقي في حذف SQLite على Windows لأن `Connection` لم يُغلق؛ أُضيف إغلاق صريح وأعيدت المجموعة بنجاح. كذلك أزيلت مشكلة إقفال ملف token cache على Windows باستخدام قراءات موضعية محدودة بدل memory map.

أمر التحقق النهائي من جذر المشروع:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

**43 passed، 0 failed خلال 4.14 ثانية** (28 اختبارًا للمرحلة الثالثة و15 للمرحلة الرابعة). لم تُشغّل CUDA أو أي تدريب.

## المشكلات والقيود المتبقية

1. ملفات TXT تُعامل كسجل لكل سطر؛ إذا احتوى TXT على نص طويل متعدد الأسطر فسيصبح عدة وثائق. يمكن إضافة اختيار لفواصل الفقرات لاحقًا.
2. ملف واحد لكل سجل JSONL؛ يجب أن يكون كل سجل JSONL سطرًا واحدًا صالحًا.
3. إذا كانت البيانات الفريدة قليلة جدًا فقد ينتج التقسيم الحتمي مجموعة فارغة؛ pipeline يرفض النتيجة ويطلب بيانات أكثر أو تغيير النسبة بدل إنشاء split غير صالح.
4. الحقول المدعومة في JSONL محصورة بـ`text` أو `content`.
5. بناء token cache يحتاج مساحة مؤقتة تقارب ضعفي عدد tokens بالبايتات، ويعيد كتابته عند إنشاء Dataset جديد.
6. حقول التدريب المستقبلية تتحقق شكليًا فقط؛ لم يبدأ التدريب ولم تُضف scheduler أو optimizer pipeline.
7. الحجم الأقصى للوثيقة قابل للضبط، لكن `ByteTokenizer.encode` ينشئ قائمة IDs للوثيقة الواحدة في الذاكرة؛ الحد الافتراضي يقيّد ذلك، ويمكن لاحقًا إضافة ترميز متدفق.

## التوافق مع المرحلة الثالثة

لم تتغير بنية `GPT`, `GPTConfig`, أو واجهة `forward` و`generate`، ولم تتغير IDs الـTokenizer. Factory ينشئ `GPTConfig` بقيم متوافقة مع الحجم 260. اختبارات المرحلة الثالثة ظلت ناجحة مع كامل اختبارات المرحلة الرابعة.

## توصيات المرحلة الخامسة

- بناء حلقة تدريب باستخدام `create_model`, `create_dataloaders`, وconfig المركزي.
- إضافة إزاحة سياقية مدروسة وتقارير loss/validation دون قراءة البيانات كلها في الذاكرة.
- تصميم checkpoint للتدريب يشمل optimizer وstep وseed مع الحفاظ على توافق تحميل checkpoint الحالي.
- إضافة قياس reproducibility، حفظ إحصاءات البيانات، والتحقق من متطلبات الترخيص قبل إدخال corpora فعلية.

## الخلاصة

اكتمل نطاق المرحلة الرابعة واختُبر عمليًا على CPU: ربط JSON بالنموذج، Model Factory، تجهيز TXT/JSONL وتقسيم deduplicated، وDataset/DataLoader. نتيجة مجموعة الاختبارات الكاملة: **43/43 ناجحة**. لم يبدأ التدريب أو اختبار GPU.
