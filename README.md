<p align="center">
  <img src="ui/logo.svg" alt="JABRIX AI Logo" width="420" />
</p>

# JABRIX AI `v3.2.0` — Local-First Multilingual Autonomous AI Platform (Arabic · English · French)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ypy999/NourAI/blob/main/notebooks/NourAI_Cloud_GPU_Training_Colab_Kaggle.ipynb)

**JABRIX AI** *(formerly NourAI — preserving 100% backward compatibility with `nourai` Python modules and `/api/v1/*`)* هو نظام ذكاء اصطناعي محلي مستقل متعدد اللغات (**العربية، الإنجليزية، الفرنسية**) يجمع بين نموذج **Decoder-only Pre-LN Transformer**، ونواة وكيل ذاتي متكاملة (`Research Agent`, `Project Manager`, `DAG Tool Orchestrator`, `4-Tier Memory`, `Knowledge Graph`, `Multimodal Document Intelligence`)، ولوحة تحكم **Control Center 2.0** (`/control-center`)، وواجهة محادثة متجاوبة (`RTL/LTR Web UI`)، ودعم التشفير (`TLS 1.3 HTTPS`).

> **تشغيل سريع:** `python run_jabrix.py` (أو `python run_nourai.py`) ← الواجهة الرئيسية `/` ولوحة التحكم `/control-center`.

---

## 1. المكونات المعمارية (`NOURAI V1.2.0`)

- **المُرمِّز النصي (`Tokenizer`):** [byte_tokenizer.py](file:///d:/NourAI/tokenizer/byte_tokenizer.py) غير فاقد (`vocab_size = 260`: المعرّفات `0..255` تمثل بايتات `UTF-8` الخام، والرموز الخاصة هي `256:<PAD>`, `257:<BOS>`, `258:<EOS>`, `259:<UNK>`).
- **معمارية النموذج (`GPT Transformer`):** [transformer.py](file:///d:/NourAI/model/transformer.py) بثلاثة أحجام مختبرة:
  - `Default / Nano`: `4` طبقات، `4` رؤوس، `d_model=128`، `context_length=128` (**`843,008` معاملاً**).
  - `Small`: `6` طبقات، `8` رؤوس، `d_model=256`، `context_length=256` (**`4,871,168` معاملاً**).
  - `Base`: `8` طبقات، `8` رؤوس، `d_model=512`، `context_length=512` (**`25,615,360` معاملاً**).
- **البيانات المتوازنة (`Corpus & SFT`):**
  - `data/processed/v1_2_corpus/splits/`: تقسيمات متوازنة على أساس عدد التوكنات/البايتات (`~33.3%` لكل من العربية والإنجليزية والفرنسية) مع صفر تسرب بين التقسيمات (`0` Leakage).
  - `data/processed/v1_2_sft/splits/`: مجموعة تعليمات ومهام متعددة (`72` تدريب / `36` تحقق / `36` اختبار) بدون أي تداخل في المطالبات أو الإجابات بين التقسيمات.
- **محرك الاستدلال والشفافية (`Inference Engine & Fallback Assistant`):**
  - الوضع الافتراضي (`use_capabilities=False`): يعتمد مباشرةً على التوليد العصبي من [engine.py](file:///d:/NourAI/inference/engine.py) (`response_source="neural_model"`) ويعرض مقاييس الأداء الحقيقية للشبكة العصبية.
  - الوضع الاحتياطي الاختياري (`use_capabilities=True` أو تمرير `"use_fallback_assistant": true` في الطلب): يستخدم المساعد القاعدي الاحتياطي [capabilities.py](file:///d:/NourAI/nourai/capabilities.py) ويُصرّح في الاستجابة والواجهة بأن المصدر هو `"rule_based_fallback_assistant"` دون خلطه بمقاييس الشبكة العصبية.
- **الواجهة البرمجية والأمان (`API, UI & Security`):**
  - [backend.py](file:///d:/NourAI/api/backend.py): مصادقة `PBKDF2-HMAC-SHA256` (`100,000` تكرار)، ومفتاح توقيع يُقرأ من متغير البيئة `NOURAI_SECRET_KEY` (أو يُولّد عشوائياً عبر `secrets.token_hex(32)` عند عدم ضبطه)، وتحديد معدل الطلبات (`Rate Limiting`) على جميع المسارات العامة والمحمية بما فيها `/api/v1/auth/guest`.
  - [index.html](file:///d:/NourAI/ui/index.html): واجهة عربية `RTL` وإنجليزية/فرنسية `LTR` تعرض مصدر الرد (`نموذج عصبي` أو `مساعد قاعدي احتياطي`) وتتيح التبديل الصريح بينهما.
  - [production.py](file:///d:/NourAI/deployment/production.py): خادم `HTTPS (TLSv1.3)` حقيقي، نسخ احتياطي واستعادة `SQLite WAL`، ومراقبة حية.

---

## 2. التشغيل والاختبار (Windows PowerShell)

```powershell
# 1. تشغيل مجموعة الاختبارات الكاملة (مع تمرير --basetemp لتجنب قيود الصلاحيات في مجلد Temp للنظام)
D:\NourAI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp="D:\NourAI\reports\pytest_tmp"

# 2. تشغيل الواجهة الرسومية والخادم المحلي (التوليد العصبي المباشر افتراضياً)
D:\NourAI\.venv\Scripts\python.exe D:\NourAI\run_nourai.py

# 3. تشغيل المحادثة التفاعلية في الطرفية (CLI)
D:\NourAI\.venv\Scripts\python.exe D:\NourAI\run_nourai.py --cli

# 4. تشغيل التدريب العصبي الموسّع وتقييم التوليد قبل/بعد
D:\NourAI\.venv\Scripts\python.exe scripts\train_and_evaluate_neural_v1_2.py

# 5. تفعيل إصدار أو التراجع الآلي (Rollback) بين v1.0.0 و v1.1.0 و v1.2.0
D:\NourAI\.venv\Scripts\python.exe scripts\rollback_release.py --version v1.2.0
```
