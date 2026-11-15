# تقرير المرحلة 10.5 — بناء محول بيانات منظمة التجارة العالمية (WTO Adapter Implementation)

## 1. الملخص التنفيذي
تم استكمال **المرحلة 10.5** بنجاح عبر بناء المحول القياسي القابل لإعادة الإنتاج `wto_en_fr_adapter_v1` للمصدر المعتمد `wto_public_corpus_en_fr` داخل [wto_adapter.py](file:///D:/NourAI/corpus/adapters/wto_adapter.py)، وربطه بخط الاستيراد القياسي في [importers.py](file:///D:/NourAI/corpus/importers.py) و[pipeline.py](file:///D:/NourAI/corpus/pipeline.py) و[reproducibility.py](file:///D:/NourAI/corpus/reproducibility.py).

بذلك أصبحت جميع المصادر الثلاثة المعتمدة (`tatoeba_cc0`, `qnl_arabic_ocr_v2`, `wto_public_corpus_en_fr`) مدعومة بالكامل (`SUPPORTED`) داخل مستودع الكود، وتُرجع دالة `build_plan()` الآن `ready: True` مع `0` معوقات (`blockers: []`).

---

## 2. المكونات البرمجية المنفذة

1. **[corpus/adapters/wto_adapter.py](file:///D:/NourAI/corpus/adapters/wto_adapter.py)**:
   - **`iter_reconstructed_wto_rows()`**: قراءة تدفقية (Streaming) لملف `Eng-Fra_UTF8/Eng-Fra-UTF8-00.txt` ذي الأعمدة العشرة (`ID, IdxID, SegID, BTYear, BTPath, BTName, SegSrc, SegTgt, Match, IdxDomain`) مع إعادة بناء الأسطر المكمّلة (Continuation Lines) التي تحتوي على فواصل أسطر داخل المقاطع النصية دون أي أخطاء تحليل (`0` أخطاء عبر `895,126` سجلاً).
   - **`scan_wto_member_domains()`**: مسح تدفقي أولي لحصر الوثائق حسب المجال (`IdxDomain`) واستبعاد الصفوف غير المزدوجة (`159,317` صفاً فارغاً في أحد الطرفين).
   - **`select_domain_stratified_documents()`**: اختيار طبقي حتمي للوثائق داخل كل مجال وفق المعادلة المعتمدة $\min(\text{available}, \max(3, \lceil 0.02 \times \text{available} \rceil))$ مرتبةً بتجزئة `SHA-256(seed:domain:IdxID)`، مما ينتج `124` وثيقة مختارة عبر `16` مجالاً في `chunk00`.
   - **`iter_wto_archive_records()`**: بث السجلات الثنائية (`eng` من `SegSrc` و`fra` من `SegTgt`) مع إرفاق:
     - `document_id = IdxID`
     - `translation_group_id = "{IdxID}:{SegID}"`
     - `source_record_id = "{ID}:eng"` و`"{ID}:fra"` (تجنباً لتصادم المعرفات عندما يتطابق النص الإنجليزي والفرنسي حرفياً مثل رموز الوثائق `G/C/W/`)
     - `wto_record_id = ID`، `domain = IdxDomain or "UNKNOWN"`، و`adapter_version = "wto_en_fr_adapter_v1"`.

2. **ربط المحول بخط الأنابيب والتدقيق**:
   - تحديث [corpus/adapters/__init__.py](file:///D:/NourAI/corpus/adapters/__init__.py) و[corpus/pipeline.py](file:///D:/NourAI/corpus/pipeline.py) و[corpus/reproducibility.py](file:///D:/NourAI/corpus/reproducibility.py).
   - إضافة اختبارات القبول في [tests/test_phase10_5_wto_adapter.py](file:///D:/NourAI/tests/test_phase10_5_wto_adapter.py).
