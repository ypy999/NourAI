# تقرير المرحلة 10.6 — استعادة التتبع والبيانات الوصفية (Provenance & Metadata Recovery)

## 1. الملخص التنفيذي
تم استكمال **المرحلة 10.6** بنجاح عبر ترقية مخطط قاعدة البيانات الوصفية في [storage_manager.py](file:///D:/NourAI/data/storage_manager.py)، ومزامنة جميع المصادر الـ `14` المسجلة في [dataset_sources.json](file:///D:/NourAI/configs/dataset_sources.json) داخل جداول `sources` و`license_current`، وفهرسة السجلات المستوردة والمقسمة كافة (`217,913` صفاً مفهرساً) داخل قاعدة البيانات المستقلة [phase10_3_corpus.sqlite3](file:///D:/NourAI/data/metadata/phase10_3_corpus.sqlite3).

---

## 2. الترقيات المنفذة

1. **ترقية جدول `records` في SQLite ([storage_manager.py](file:///D:/NourAI/data/storage_manager.py))**:
   - إضافة أعمدة التتبع الصريحة مع دعم الترقية التلقائية (`_ensure_records_schema`):
     - `document_id`
     - `translation_group_id`
     - `domain`
     - `ingestion_timestamp`
     - `content_hash`
   - إنشاء الفهارس المركبة `idx_records_dataset_document`، و`idx_records_dataset_tgroup`، و`idx_records_dataset_domain`.
   - إضافة دالة `sync_catalog()` لمزامنة جميع مصادر الكتالوج الـ `14` تلقائياً، ودالة `provenance_summary()` لإصدار تقارير التغطية.

2. **الحفاظ على تجزئة النص الأصلي (`raw_text_sha256`) عبر التطبيع المدمج والتقسيم ([processor.py](file:///D:/NourAI/data/processor.py))**:
   - تحديث `DataPipeline._iter_records()` للحفاظ على `raw_text_sha256` و`normalized_text_sha256` دون إعادة تضخيم النصوص المرحلية (`raw_text`, `normalized_text`, `cleaned_text`) عند قراءة ملفات التطبيع المدمج (`retain_stage_texts=False`).

---

## 3. إحصائيات الفهرسة والتغطية (`multilingual:phase10_3`)

| المؤشر | القيمة | نسبة التغطية |
| --- | --- | --- |
| إجمالي المصادر المتزامنة في الكتالوج (`sources` & `license_current`) | `14` | `100%` |
| إجمالي السجلات المفهرسة في `multilingual:phase10_3` | `103,670` | `100%` |
| سجلات تحتوي على `source_record_id` | `103,670` | `100%` |
| سجلات تحتوي على `ingestion_timestamp` | `103,670` | `100%` |
| سجلات تحتوي على `content_hash` | `103,670` | `100%` |
| سجلات QNL + WTO التي تحتوي على `document_id` | `47,954` (`58` QNL + `47,896` WTO) | `100%` |
| سجلات WTO التي تحتوي على `translation_group_id` و`domain` | `47,896` | `100%` |
