# تقرير المرحلة 10.9 — تدقيق تسرب الوثائق والتشابه (Document Leakage & Similarity Audit)

## 1. الملخص التنفيذي
تم استكمال **المرحلة 10.9** بنجاح عبر بناء مدقق التسرب والتشابه الشامل في [leakage_audit.py](file:///D:/NourAI/corpus/leakage_audit.py) وتشغيله على التقسيمات المعتمدة في [data/processed/phase10_3/splits/](file:///D:/NourAI/data/processed/phase10_3/splits) (`train.jsonl`, `validation.jsonl`, `test.jsonl`).

أثبت التدقيق عدم وجود أي تسرب نصي متطابق أو مطبّع، وعدم وجود أي تسرب في معرفات الوثائق (`document_id`) أو مجموعات الترجمة الثنائية (`translation_group_id`) بين التقسيمات الثلاثة.

---

## 2. نتائج فحص التسرب والتشابه

| معيار التدقيق | القيمة المسجلة | الحالة |
| --- | --- | --- |
| تسرب التجزئة المتطابقة (`exact_hash_leakage`) | `0` | `PASS` |
| تسرب التجزئة المطبّعة (`normalized_hash_leakage`) | `0` | `PASS` |
| تسرب الوثائق الصريحة (`document_id_leakage`) | `0` | `PASS` |
| تسرب أزواج الترجمة الثنائية (`translation_group_leakage`) | `0` | `PASS` |
| تسرب مفاتيح التجميع (`split_group_id_leakage`) | `0` | `PASS` |
| فحص بصمات التشابه القريب (`5-word min-hash shingles`) | مكتمل وموثق في [phase10_9_leakage_audit_report.json](file:///D:/NourAI/reports/phase10/phase10_9_leakage_audit_report.json) | `PASS` |
