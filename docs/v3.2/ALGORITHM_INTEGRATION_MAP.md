# خريطة دمج منظومة الـ150 خوارزمية داخل مشروع JABRIX AI (`ALGORITHM_INTEGRATION_MAP.md`)

تم تطبيق قواعد الدمج الهندسي الصارمة (عدم تشغيل كل الخوارزميات في كل مرة، احترام قيود العتاد `Intel Core i3 / 8 GB RAM / CPU-only`، توفير واجهة موحدة، تفعيل `LRU Cache`، وقياس التحسن فعلياً) عبر بناء وحدة تشغيلية حقيقية في [nourai/algorithm_hub/](file:///d:/NourAI/nourai/algorithm_hub/) وربطها بالـ API (`/api/v3/algorithms/catalog` و`/api/v3/algorithms/run`).

---

## 1. الخوارزميات التي يحتاجها مشروع JABRIX AI فعلاً (تصنيف الـ15 محركاً)

| المحرك | موجود أصلاً في JABRIX AI | تم دمجه فعلياً في `algorithm_hub` | مؤجّل / غير مناسب لـ 8GB CPU-only |
| :--- | :--- | :--- | :--- |
| **1. الذكاء الاصطناعي** | `#1 Transformer`, `#3 GPT` | — (نموذج واحد كافٍ محلياً) | `BERT, T5, RoBERTa, ALBERT, ELECTRA, DeBERTa, XLNet, MoE` (تستهلك RAM/GPU مفرطاً) |
| **2. الفهم اللغوي** | `#14 TF-IDF`, `#19 Byte-Level BPE` | **`#15 BM25`**, **`#16 TextRank`** | `Word2Vec, GloVe, FastText, CRF, HMM, SentencePiece` |
| **3. التعلم العميق** | — | — | `CNN, RNN, LSTM, GRU, ResNet, DenseNet, EfficientNet, VAE, U-Net` (خارج نطاق النص الحالي) |
| **4. تحليل البيانات** | — | — | `XGBoost, LightGBM, CatBoost, RF, SVM` (تُضاف عند وجود مهام جدولية/ML مخصصة) |
| **5. البحث والاسترجاع** | `SQLite` / `TF-IDF` | **`#42 Dijkstra`**, **`#43 BFS`** (بحث متعدد القفزات على KnowledgeGraph) | `HNSW, LSH` (عند تجاوز 100K وثيقة) |
| **6. الذاكرة والتخزين** | `SQLite WAL`, `4-Tier Memory` | **`#51 LRU Cache`**, **`#56 Bloom Filter`**, **`#59 Merkle Tree`** | `ARC, Consistent Hashing, Count-Min, HyperLogLog` |
| **7. التحسين الرياضي** | `#64 AdamW`, Cosine LR, L2 Anchor | — | `Genetic, PSO, Newton` |
| **8. التنبؤ الزمني** | — | **`#73 Exponential Smoothing`**, **`#74 Kalman Filter`** (تصفية قياسات RAM/Latency) | `ARIMA, SARIMA, Prophet, TCN` |
| **9. التعلم المعزز** | — | — | `DQN, PPO, A2C, SAC, TD3, MCTS` (مؤجّلة لمرحلة RLHF/GPU) |
| **10. الرؤية الحاسوبية** | فحص ترويسة الصور + كشف OCR | — | `ViT, YOLO, Faster/Mask R-CNN, MobileNet` (تتطلب GPU/أوزان رؤية) |
| **11. معالجة النصوص** | تطبيع عربي/فرنسي + Deduplication | **`#101 Levenshtein`**, **`#102 Jaro-Winkler`**, **`#104 MinHash`** | `K-Means, DBSCAN, PCA, t-SNE, UMAP` |
| **12. الأمان والتشفير** | `#116 SHA-256`, `#119 HMAC`, `PBKDF2`, `TLS 1.3` | **`#120 HKDF-SHA256`** | `AES-256, ChaCha20, RSA-OAEP, Ed25519, Argon2id` (تتطلب مكتبة `cryptography` اختيارية) |
| **13. المعالجة المتوازية** | `DAGToolExecutor` (`ThreadPoolExecutor`) | **`#121 MapReduce`** + تنفيذ متوازٍ في `AlgorithmOrchestrator` | `SIMD, GPU Parallel Reduction` |
| **14. الضغط والتخزين** | `zlib` (`FlateDecode`) | **`#139 RLE`**, **`#140 Delta Encoding`** | `Arithmetic, Brotli, Zstd, LZ4` |
| **15. التقييم والتحقق** | بوابة الانحدار `check_regression_gate` | **`#143 Bootstrap CI`**, **`#148 F1 Score`**, **`#150 Ensemble Fusion`** | `Bayesian Opt, Grid/Random Search` |

---

## 2. الخوارزميات التي تعمل بالتوازي
داخل [core.py](file:///d:/NourAI/nourai/algorithm_hub/core.py) (`AlgorithmOrchestrator`)، تعمل الخوارزميات المستقلة بالتوازي عبر `ThreadPoolExecutor`:
- مسار البحث والترتيب (`search_rank`): `BM25` ∥ `TextRank`
- مسار التشابه الضبابي (`text_similarity`): `Levenshtein` ∥ `Jaro-Winkler` ∥ `MinHash`
- مسار البحث على الرسم البياني (`graph_search`): `BFS` ∥ `Dijkstra`
- مسار الأمان والنزاهة (`integrity_security`): `Merkle Tree` ∥ `Bloom Filter` ∥ `HKDF-SHA256`
- مسار التنبؤ بالعتاد (`forecast_telemetry`): `Kalman Filter` ∥ `Exponential Smoothing`

## 3. الخوارزميات التي تحسّن أو تحل محل أجزاء أبطأ
- **`LRU Cache (#51)`:** يختصر زمن الطلبات المتكررة إلى جزء من الميلي ثانية (`cache_hit=True`).
- **`Bloom Filter (#56)`:** فحص عضوية فوري `O(k)` قبل الاستعلام عن قاعدة البيانات.
- **`BM25 (#15)`:** ترتيب أدق للوثائق الطويلة والقصيرة مقارنةً بـ `TF-IDF` البسيط.

## 4. القدرات الجديدة المضافة للفهم والتحليل
- استخلاص الكلمات المفتاحية والملخصات بالرسم البياني (`TextRank`).
- البحث متعدد القفزات الموزون على `KnowledgeGraph` (`BFS` و`Dijkstra`).
- مطابقة الاستعلامات العربية/الفرنسية/الإنجليزية رغم الأخطاء الإملائية (`Levenshtein` + `Jaro-Winkler`) وكشف شبه التكرار (`MinHash`).
- جذر `Merkle Tree` لضمان عدم العبث بسجلات التدقيق والذاكرة، و`Kalman Filter` لتنعيم قراءات زمن الاستجابة و`RAM`، و`Bootstrap 95% CI` لتقييم الثقة الإحصائية.

## 5. التحقق والاختبار
تم التحقق عبر [test_algorithm_hub.py](file:///d:/NourAI/tests/test_algorithm_hub.py) (اختبارات الوحدة، التوازي، الدمج وكشف التعارض، تسريع `LRU Cache`، وتكامل `/api/v3/algorithms/catalog` و`/api/v3/algorithms/run`).
