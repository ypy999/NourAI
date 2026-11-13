# NourAI — Phase 10 Final Closure

**الحالة: PHASE_10_PARTIALLY_COMPLETE**  
وقت التنفيذ: 2026-11-13T01:50:51+01:00  
المشروع: D:\NourAI  
القرار: Phase 10 غير معتمدة مكتملة؛ الانتقال إلى Phase 11 غير مسموح.

## ما اكتمل
- ربط 55,716 صف split بمصدره عبر المسار ورقم السطر الصريحين، ثم التحقق من source_record_id والنص normalized. أُنشئت نسخة إضافية في splits_v2_verified؛ ملفات Phase 8 بقيت كما هي.
- source ID وExact text وNormalized text leakage بين التقسيمات: صفر.
- فحص exact/normalized/LSH شمل 55,719 سجلًا دون حذف.
- راجعنا 12 مدخلاً في Source Registry، وأجرينا قياسات CPU كاملة للـTokenizer وDataset وDataLoader.
- أصلحنا حواجز المسارات في RegistrySync وDownloadManager، ومنعنا تحويل رقم السطر إلى source_record_id عند غياب المعرف.
- نجحت الاختبارات غير التدريبية بعد آخر التعديلات.

## البيانات وprovenance
- المطابقة السابقة Raw→Imported أثبتت 55,719 سجلًا مطابقًا عبر source_record_id؛ لم نكررها في هذا الإغلاق.
- Imported وNormalized يحتويان 55,719 سجلًا. Splits: Train 50,144، Validation 2,787، Test 2,785.
- وثيقة leakage: NOT_CHECKABLE؛ لا توجد document IDs في بيانات Tatoeba الجملية. لم نختلق IDs.
- LSH أظهر 50 زوجًا محتملًا، منها 12 زوجًا عابرًا للتقسيم؛ مراجعتها البشرية ما زالت معلقة. LSH ليس مقارنة شاملة لكل زوج.
- SQLite للقراءة فقط: integrity_check=ok، foreign_key_check بلا مخالفات، 111,435 صف metadata.
- **تنبيه:** قاعدة SQLite القديمة تحتوي 55,716 صف Split بقيم مولّدة مثل train:1 في source_record_id. لم نعدّلها؛ الكود الجديد يخزن NULL عند غياب المعرف. يلزم إجراء منفصل لإصلاح السجلات التاريخية.
- بصمات 10/10 ملفات البيانات والإعدادات من baseline السابق ما زالت مطابقة.

## الترخيص وجاهزية اللغات
راجعنا 12 مصادر: 1 VERIFIED، 7 RESTRICTED، 4 UNVERIFIED، 0 REJECTED. المصدر المستخدم فقط هو tatoeba_cc0؛ لم نستورد غيره. المراجعة الهندسية ليست رأيًا قانونيًا.

| اللغة | السجلات | Train | Validation | Test | الحالة |
|---|---:|---:|---:|---:|---|
| العربية | 2 | 2 | 0 | 0 | NOT_READY |
| الإنجليزية | 41,515 | 37,380 | 2,070 | 2,062 | NOT_READY |
| الفرنسية | 14,202 | 12,762 | 717 | 723 | NOT_READY |

العربية بلا validation/test. الإنجليزية والفرنسية لديهما مصدر واحد ولا توجد مجالات موثقة حسب السياسة. المؤشرات heuristic ولا تقيس الفهم أو صحة المعنى.

## الأداء والتكامل على CPU
- Python 3.15.2، PyTorch 2.15.0+cpu، CPU فقط؛ لم تستخدم CUDA.
- ByteTokenizer عالج 55,719 سجلًا و4,305,623 token في 1.355s، بمعدل 3,177,494 token/s؛ أخطاء الصفوف/IDs/round-trip صفر.
- DataLoader بحجم batch 8 وworkers 0: Train 1,049,365، Validation 1,042,147، Test 1,078,815 token/s.
- اختبارات الأبعاد والأنواع وإزاحة next-token وثبات shuffle نجحت. التحميل يستخدم fixed windows بلا padding أو attention-mask output ويسقط الجزء الختامي الأقصر من نافذة كاملة.
- Peak working set بعد القياس 212.6 MiB. هذه سرعة إدخال بيانات وليست سرعة تدريب.

## الاختبارات

| النطاق | ناجح | فاشل | متجاوز | الزمن |
|---|---:|---:|---:|---:|
| مجموعة الاختبارات غير التدريبية | 156 | 0 | 0 | 11.54s |
| الاختبارات المركزة بعد إصلاح الأمان | 33 | 0 | 0 | 0.94s |

استثنينا tests/test_training.py وعددها 21 اختبارًا لأنها تتضمن Trainer.fit وتحديث الأوزان؛ جمعناها فقط ولم نشغلها.

## الملفات المعدلة
- data/storage_manager.py: عدم اختلاق معرف المصدر.
- data/registry_sync.py: حصر مسارات DB/registry في جذر المشروع.
- data/download_manager.py: حصر مسارات raw وmetadata في جذر المشروع.
- أضيفت اختبارات regression للملفات الثلاثة. نسخ ما قبل التعديل موجودة في .backup/phase10-final-closure وبصماتها في sha256_manifest.json.

## العوائق والقرار
1. معرفات Split القديمة المصطنعة في SQLite لم تُصلح.
2. Document leakage لا يمكن فحصه من البيانات المتاحة.
3. 12 مرشح near-duplicate عابر للتقسيم ينتظر قرارًا بشريًا.
4. اللغات الثلاث NOT_READY.
5. pipeline_version وdocument_id وبعض حقول الإصدارات غير متوفرة تاريخيًا.

**القرار النهائي: PHASE_10_PARTIALLY_COMPLETE. لا تبدأ التدريب أو Phase 11.** نجاح الاختبارات لا يثبت جودة اللغة أو اكتمال القانونية أو خلو التسرب.

## الأدلة
كل تقارير الإغلاق في D:\NourAI\reports\phase10\final_closure_v2: provenance_final.json، split_integrity_final.json، document_leakage_final.json، near_duplicate_audit.json، legal_review_final.json، performance_final.json، security_final.json، dataset_dataloader_final.json، language_readiness_final.json، tests_final.json، blockers_final.json، phase10_closure.json، sha256_manifest.json، وFINAL_REPORT.md.
