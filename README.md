# Nūr AI — local prototype

مرحلة أولى قابلة للتشغيل على CPU: نموذج decoder-only صغير من PyTorch، tokenizer محلي قائم على بايتات UTF-8 (يغطي العربية والفرنسية والإنجليزية بلا ملف مفردات خارجي)، تدريب تجريبي، توليد نص وواجهة FastAPI اختيارية. لا توجد أوزان مدربة مسبقًا ولا اتصال بنموذج خارجي.

## المتطلبات والتشغيل (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\train.py --text data\sample.txt --steps 100
python scripts\chat.py
```

ابدأ بتجربة صغيرة؛ ناتج التدريب على العينة للتأكد من أن المسار يعمل فقط ولا يمثل مساعدًا ذا قدرات عامة. استبدل `data/sample.txt` بنصوص تملك حق استخدامها. حجم النموذج الافتراضي نحو 0.5 مليون معامل، سياق 128، ودفعة 8، على CPU.

لواجهة API: `uvicorn api.app:app --host 127.0.0.1 --port 8000` بعد وجود `checkpoints/latest.pt`.

## المكونات

- `model/transformer.py`: decoder Transformer مع causal attention.
- `tokenizer/byte_tokenizer.py`: تحويل UTF-8 bytes إلى IDs قابلة للعكس (256 بايت + 4 رموز خاصة).
- `training/`: batches، تدريب، تحقق، checkpoint قابل للاستئناف.
- `chat/engine.py`, `database/store.py`: توليد محلي وسجل SQLite.
- `api/app.py`: endpoints محلية أساسية.
- `tests/`: اختبارات وحدات أساسية.

## ملاحظة حول gpt_vocab.zip

يضم الأرشيف صيغ BPE مختلفة (`encoder.json` مع `vocab.bpe`، وملفات tiktoken متعددة). لم نخلطها أو نستوردها تلقائيًا؛ يلزم التحقق من كل زوج وترخيصه وتغطية العربية قبل اختيار واحد. tokenizer البايتات الحالي بديل أولي بسيط وليس BPE.
