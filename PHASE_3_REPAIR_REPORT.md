# تقرير إصلاح المرحلة الثالثة — NourAI

## النطاق والبيئة

تمت قراءة `PHASE_3_TRANSFORMER_AUDIT.md` بالكامل قبل التعديل. اقتصر العمل على Transformer وTokenizer والاختبارات، ولم يبدأ تدريب أو API أو واجهة. بقيت Python الأساسية `3.15.2` كما هي. أُنشئت نسخة احتياطية قبل التغيير في `.backup/phase3-before-fix/` للملفات `transformer.py`, `byte_tokenizer.py`, و`default.json`.

اختيرت بيئة افتراضية مستقلة في `.venv`. PyTorch 2.14.0 لم يكن متاحًا لهذا المفسر من فهرس الحزم؛ توفر wheel Windows `torch 2.15.0` لـCPython 3.15، وتم تثبيته مع pytest وNumPy داخل `.venv` فقط. لا توجد بطاقة NVIDIA ظاهرة في فحص الجهاز السابق؛ الاختبارات نُفذت على CPU. مصدر التوافق: [صفحة إصدارات PyTorch السابقة](https://docs.pytorch.org/get-started/previous-versions/) ونتيجة الفهرس أثناء التثبيت.

## الملفات التي تغيرت

| الملف | التعديل | السبب |
|---|---|---|
| `model/transformer.py` | تحقق GPTConfig والمدخلات، فصل causal mask، تحسين generation مع EOS والتحقق واستعادة وضع التدريب، وإضافة save/load checkpoint آمن باستخدام state_dict/config | منع حالات غير صالحة وإتاحة اختبار السلوك والحفظ والاستعادة |
| `tokenizer/byte_tokenizer.py` | تحقق نوع النص وIDs، سياسة فك UTF-8 اختيارية strict/replace/ignore، رفض IDs خارج المفردات، مع الحفاظ على تجاهل الرموز الخاصة عند decode | كشف فساد البيانات مع بقاء السلوك العام المعتاد |
| `tests/test_phase3.py` | 28 اختبارًا للوحدات والتكامل الصغير | إثبات الإصلاحات على PyTorch فعلي |
| `.venv/` | بيئة مستقلة تحتوي torch 2.15.0 وpytest وNumPy | تشغيل الاختبارات دون لمس Python الأساسية |
| `.backup/phase3-before-fix/` | نسخ الملفات الأصلية المستهدفة | استعادة النسخة السابقة عند الحاجة |

لم تتغير واجهة `GPT.forward`، ولا توقيع `generate` الحالي إلا بإضافة معامل اختياري `eos_id`. بقيت الإعدادات الافتراضية كما كانت. لم يتم ربط `configs/default.json` بالمنشئ في هذه المرحلة؛ هذا يحتاج قرار تصميم/ربط أوسع واختبار تحميل إعدادات.

## تفاصيل الإصلاح

### Transformer

- أضيف تحقق للأعداد الموجبة، توافق `d_model % n_heads == 0`، وحدود dropout.
- جرى فصل القناع في `causal_mask`; القيمة `True` تحجب المستقبل وفق دلالة PyTorch. اختبار سلوكي غيّر التمثيلات المستقبلية وأثبت أن مخرجات المواضع السابقة لا تتغير.
- يرفض `forward` المدخلات غير ثنائية الأبعاد أو الفارغة أو ذات IDs خارج المفردات، ويتحقق من dtype وشكل ومدى targets.
- loss ما زال Cross Entropy على logits المسطحة وtargets. الاختبار تحقق من scalar finite ومن مرور gradients غير صفرية إلى embedding وattention وFFN.
- يتحقق `generate` من prompt والخيارات، ويوقف كل العينة عند توليد EOS، ويحافظ على السلوك السابق بعدد أمثلة batch عبر استكمال EOS للمنتهية، ثم يعيد النموذج لوضع التدريب السابق.
- أضيف `save_checkpoint` و`load_checkpoint` لتخزين config وstate_dict مع `format_version`. التحميل يستخدم `weights_only=True` عندما يدعمه PyTorch، ويتحقق من البنية عبر تحميل strict.

### Tokenizer

- يظل UTF-8 byte tokenizer بحجم 260، مع IDs خاصة 256–259.
- `encode` يرفض غير النص ويستخدم UTF-8 strict.
- `decode` يقبل integers فقط، ويرفض IDs خارج 0–259، ويتيح سياسة UTF-8 صريحة. السياسة الافتراضية `replace` محافظة ومتوافقة مع السلوك السابق؛ يمكن استخدام `strict` لكشف أجزاء UTF-8 المبتورة.
- لا يضيف الاختبار أو الإصلاح BPE أو padding فعليًا أو تخزين حالة streaming.

## الاختبارات الفعلية

الأمر المنفذ من جذر `D:\NourAI`:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

**النتيجة النهائية: 28 passed خلال 3.13 ثانية، دون تحذيرات.**

الاختبارات تشمل: توافق الحجم مع config/tokenizer، configs غير الصالحة، مصفوفة القناع، عدم تسرب المستقبل، shapes، loss، backward وgradient flow، رفض IDs والأطوال/targets غير الصحيحة، EOS stopping، خيارات generation وprompt الفارغ، round-trip عربي/إنجليزي/فرنسي/emoji/NUL، الرموز الخاصة والـIDs غير الصالحة وUTF-8 المبتور، ثم save/load ومقارنة logits.

كان التشغيل الأولي `28 passed` مع تحذيرين بسبب غياب NumPy في venv ومنع كتابة pytest cache. أضيف NumPy إلى venv وأعيد التشغيل مع تعطيل cache؛ التشغيل النهائي نظيف.

## المشكلات المتبقية

1. `default.json` غير مربوط تلقائيًا بإنشاء النموذج؛ يجب إنشاء loader ومقارنة كل إعدادات الملف مستقبلًا.
2. لا توجد سياسة PAD أو masking لـPAD في loss، ولا حاجة لها ما دام batching غير مبني بعد.
3. `generate` لا يدعم top-p أو repetition penalty أو KV cache؛ لا تدخل ضمن الإصلاحات المعتمدة هنا.
4. فك UTF-8 أثناء بث tokens على أجزاء غير مكتمل؛ decode كامل السلسلة بعد التوليد يعالج الأحرف الصحيحة، أما streaming فيحتاج incremental decoder.
5. لم يُختبر CUDA، ولم يبدأ تدريب أو قياس سرعة/جودة النموذج.
6. لا يتضمن التقرير ادعاءً بأن المخرجات ذات جودة مساعد عام؛ هذه النواة ما زالت نموذجًا عشوائي الأوزان قبل التدريب.

## الحالة

أُصلحت النواقص المحددة في نطاق Transformer وTokenizer والحفظ/الاستعادة، واختُبرت فعليًا محليًا على CPU: **28/28 اختبارًا ناجحًا**. لا يلزم إجراء آخر لإغلاق هذا النطاق؛ الخطوة التالية خارج هذه الموافقة هي ربط إعدادات JSON أو بدء المرحلة التالية، وتحتاج توجيهًا منفصلًا.
