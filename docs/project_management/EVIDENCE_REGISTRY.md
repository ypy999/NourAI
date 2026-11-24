# Jebrex AI — Evidence Registry (سجل الأدلة النهائي القابل للفحص)

| معرف الدليل | التاريخ | الوصف | المسار / الأمر | النتيجة المتحقَّق منها |
|---|---|---|---|---|
| `EV-001` | `2026-11-24` | جرد العتاد ومساحة التخزين | `Get-CimInstance` + `Get-PSDrive` | `i3-10110Y` (2C/4T), `7.89 GiB RAM`, `CUDA=False`, `C: 6.95 GiB` free |
| `EV-002` | `2026-11-24` | فحص البيئة ونقاط الحفظ الـ 19 | `pip list` + `torch.load(..., weights_only=True)` | `Python 3.15.2`, `torch 2.15.0+cpu`; `nourai_v1_2_model.pt` (`4.87M`, `18.61 MiB`) |
| `EV-003` | `2026-11-24` | النسخ الاحتياطي الإلزامي (القاعدة 7) | `backups/pre_jebrex_20261124/` | نسخ احتياطية كاملة لـ 8 ملفات أصلية قبل أي تعديل |
| `EV-004` | `2026-11-24` | بوابة البيانات وتنظيف `jebrex_v1_clean` | `python -m data.jebrex_data_gate` + `pytest tests/test_jebrex_data_gate.py` | `6/6 PASS`; `jebrex_v1_clean`: `DATA_READY` (تحقق) / `DATA_NOT_READY` (إنتاج)، `0%` تشويش، `0` تسرب |
| `EV-005` | `2026-11-24` | التدريب التحققي القابل للاستئناف | `python -m training.jebrex_train` + `reports/training/jebrex_v1_sanity_report.json` | `60` خطوة (`65.73s`)؛ استعادة `AdamW` و`Scheduler` مطابقة حرفياً عند خطوة `30`؛ `Test Loss: 5.574 -> 3.224` (`PPL: 263.5 -> 25.1`) |
| `EV-006` | `2026-11-24` | اختبارات النموذج والتدريب والاستدلال | `pytest tests/test_jebrex_model_and_training.py tests/test_jebrex_inference_and_chat.py` | `9/9 PASS` (شاملاً `ignore_index`, `resume_with_recovery`, `stop_sequences holdback`, `top_k_top_p`, `ConversationContextManager`) |
| `EV-007` | `2026-11-24` | المراجعة الأمنية لـ 12 مجالاً | `python -m security.security_audit_v2` + `reports/security/security_audit_report.json` | `PASS_NO_OPEN_CRITICAL_OR_HIGH` (`17/17 PASS`, `0 CRITICAL`, `0 HIGH`, `3 MEDIUM`, `1 LOW`) |
| `EV-008` | `2026-11-24` | اختبار تطبيق Windows ومشروع Android وOpenAPI | `pytest tests/test_jebrex_apps.py` | `5/5 PASS` (اتصال HTTP حي متعدد الأدوار مع `neural_model`) |
| `EV-009` | `2026-11-24` | المعيار المقارن الموحد لـ 12 مجالاً (`21` سؤالاً) | `python -m evaluation.jebrex_benchmark` + `reports/final_completion/BENCHMARK_RESULTS.json` | `v1.2 Small`: `0/21` (`PPL=809.04` نسيان كارثي)؛ `v1.2 Pretrain Nano`: `0/21` بشرياً (`1/21` إيجابي كاذب آلياً)؛ `Sanity Nano`: `0/21` |
| `EV-010` | `2026-11-24` | التشغيل الشامل النهائي لحزمة Pytest | `pytest -q -p no:cacheprovider` | **`276 passed`** (`250` قديم + `26` جديد؛ صفر تراجعات)، `11 failed` (أرشيفات `phase10_3` المستثناة في `.gitignore`)، `1 skipped` |
