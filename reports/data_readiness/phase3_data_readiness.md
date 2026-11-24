# المرحلة الثالثة — تقرير جاهزية البيانات وبوابات اللغات (`Phase 3 Data Readiness`)

- **التوقيت (UTC):** `2026-11-24T21:25:11Z`
- **بوابة التحقق المخبري قبل التنظيف (`v1_2_corpus`):** `DATA_NOT_READY`
- **بوابة التحقق المخبري بعد التنظيف (`jebrex_v1_clean`):** `DATA_READY`
- **بوابة الإنتاج لنموذج لغوي عام (`General LLM Production Gate`):** `DATA_NOT_READY`

## 1. بوابات اللغات المستقلة — قبل التنظيف (`v1_2_corpus`)

| اللغة | بوابة التحقق المخبري | بوابة الإنتاج العام | توكنات التدريب | نسبة التشويش | نقاء الخط | سجلات بلا ترخيص | أسباب المنع/المراجعة |
|:---:|:---:|:---:|---:|---:|---:|---:|---|
| `ar` | `DATA_NOT_READY` | `DATA_NOT_READY` | 45,289 | 27.8% | 0.882 | 223 | cross_split_contamination=1; noise_rate=0.278 > 0.05; avg_target_script=0.882 < 0.92; missing_per_record_license_metadata=223; intra_split_duplicates(exact=0,norm=2) |
| `en` | `DATA_NOT_READY` | `DATA_NOT_READY` | 45,023 | 3.4% | 1.000 | 676 | cross_split_contamination=1; missing_per_record_license_metadata=676; intra_split_duplicates(exact=0,norm=2) |
| `fr` | `DATA_NOT_READY` | `DATA_NOT_READY` | 45,063 | 0.1% | 1.000 | 889 | cross_split_contamination=1; missing_per_record_license_metadata=889; intra_split_duplicates(exact=0,norm=2) |

## 2. بوابات اللغات المستقلة — بعد التنظيف والتوثيق (`jebrex_v1_clean`)

| اللغة | بوابة التحقق المخبري | بوابة الإنتاج العام | توكنات التدريب | توكنات التحقق | توكنات الاختبار | نسبة التشويش | نقاء الخط | التنوع المعجمي TTR |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| `ar` | `DATA_READY` | `DATA_NOT_READY` | 31,701 | 8,184 | 8,505 | 0.00% | 0.9993 | 0.5894 |
| `en` | `DATA_READY` | `DATA_NOT_READY` | 44,537 | 9,019 | 8,169 | 0.00% | 1.0000 | 0.3240 |
| `fr` | `DATA_READY` | `DATA_NOT_READY` | 45,046 | 8,768 | 9,059 | 0.00% | 1.0000 | 0.3267 |

## 3. كشف التلوث والتسرب بين التقسيمات (`jebrex_v1_clean`)

- تسرب التطابق الدقيق (`train/val/test`): `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- تسرب التطابق المطبَّع (`Normalized`): `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- تداخل معرفات المستندات (`parent_doc_id`): `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- تداخل شبه المكرر (`5-gram Jaccard >= 0.85`): `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- إجمالي التلوث العابر للتقسيمات: **`0`**

## 4. إحصاءات التنظيف وإزالة التشويش

- تقسيمات `jebrex_v1_clean`: `{'test': {'input_records': 274, 'kept_records': 252, 'dropped_records': 22}, 'validation': {'input_records': 218, 'kept_records': 213, 'dropped_records': 5}, 'train': {'input_records': 1296, 'kept_records': 1234, 'dropped_records': 62}}`
- أسباب الاستبعاد: `{'ar:low_arabic_script_ratio': 42, 'ar:latin_ocr_contamination': 55, 'ar:excessive_punctuation_ocr_noise': 28, 'ar:ocr_punctuation_artifacts': 20, 'en:excessive_punctuation': 23, 'ar:fragmented_ocr_tokens': 12, 'fr:excessive_punctuation': 1, 'near_duplicate': 3}`
- إصلاحات SFT النحوية (`jebrex_v1_clean_sft`): `35` سجلاً مصححاً، التقسيمات: `{'test': 34, 'validation': 34, 'train': 68}`
