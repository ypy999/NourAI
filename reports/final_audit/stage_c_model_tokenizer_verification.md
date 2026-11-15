# Stage C — تدقيق المُرمِّز النصي ومعمارية النموذج (`TOKENIZER_VERIFIED` & `MODEL_VERIFIED`)

- **حالة المُرمِّز:** `TOKENIZER_VERIFIED`
- **حالة المعمارية:** `MODEL_ARCHITECTURE_VERIFIED` (`MODEL_VERIFIED`)

## 1. كفاءة ودقة `ByteTokenizer` (`vocab_size=260`)

| الفئة | تطابق الاستعادة (`Exact Match`) | عدد الأحرف | عدد التوكنات | التوكنات لكل حرف |
|---|:---:|---:|---:|---:|
| `arabic` | `True` | 119 | 220 | 1.8487 |
| `english` | `True` | 94 | 94 | 1.0 |
| `french` | `True` | 101 | 109 | 1.0792 |
| `mixed` | `True` | 75 | 87 | 1.16 |
| `punctuation` | `True` | 55 | 75 | 1.3636 |
| `numbers` | `True` | 44 | 54 | 1.2273 |
| `special_symbols` | `True` | 24 | 34 | 1.4167 |
| `long_text` | `True` | 1479 | 2679 | 1.8114 |

## 2. أحجام معمارية `GPT` المختبرة

| الحجم | الطبقات | الرؤوس | الأبعاد `d_model` | طول السياق | عدد المعاملات | القناع السببي | حفظ واستعادة |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| `Default` | 4 | 4 | 128 | 128 | `843,008` | `True` | `True` |
| `Small` | 6 | 8 | 256 | 256 | `4,871,168` | `True` | `True` |
| `Base` | 8 | 8 | 512 | 512 | `25,615,360` | `True` | `True` |