# تقرير قياس الأداء قبل وبعد التحسينات — مشروع NourAI (`PERFORMANCE_AUDIT.md`)

- **تاريخ القياس:** 2026-11-17
- **النموذج المقاس:** `checkpoints/v1_2/nourai_v1_2_model.pt` (`843,008` معامل، `CPU`, `greedy`, `max_new_tokens=48`)
- **الموجه المرجعي:** `"<|user|>\nWhat is artificial intelligence and how does NourAI work locally?\n<|assistant|>\n"`

---

## 1. مقارنة أداء التوليد (قبل vs بعد KV-Cache)

| المقياس | قبل التحسين (بدون KV-Cache) | بعد التحسين (مع KV-Cache) | نسبة التحسن |
|---|---:|---:|---:|
| **عدد التوكنات المولدة** | `48` | `48` | متطابق |
| **الزمن الكلي (`ms`)** | `1194.08 ms` | `269.74 ms` | **أسرع بـ 4.43x** |
| **معدل الإنتاج (`tokens/s`)** | `40.22 tok/s` | `178.72 tok/s` | **+344%** |
| **أول 10 معرفات توكنات** | `[66, 111, 102, 32, 105, 115, 117, 99, 101, 32]` | `[66, 111, 102, 32, 105, 115, 117, 99, 101, 32]` | **تطابق تام 100%** |

---

## 2. اختبار الضغط المتزامن (`Phase 20 Stress Benchmark` بعد التعديلات)
- **الإعداد:** `4` خيوط متزامنة (`ThreadPoolExecutor`)، `16` طلب `/api/v1/chat`
- **نسبة النجاح:** `100%` (`1.0`)
- **الكمون `p50` / `p95` / `max`:** `117.89 ms` / `136.72 ms` / `136.72 ms`

---

## 3. تحسينات قاعدة البيانات SQLite
- إزالة استعلام N+1 عند تحميل الجلسات (`SessionManager._load_from_db`) واستبداله باستعلام واحد مع تطبيق حد `max_turns_per_session * 2`.
- إضافة 6 فهارس استعلام (`idx_prod_sessions_user_id`, `idx_prod_messages_session_id`, `idx_prod_user_memories_user_id`, `idx_prod_rag_docs_user_id`, `idx_prod_rag_chunks_user_id`, `idx_prod_metrics_name_time`).
