# تقرير تدقيق المكتبات والاعتماديات — مشروع NourAI (`DEPENDENCY_AUDIT.md`)

- **نطاق الفحص:** `pyproject.toml`، `requirements.txt`، `requirements/base.txt`، `requirements/dev.txt`، `Dockerfile`، `render.yaml`، وحزم البيئة الافتراضية `.venv` (`pip list`).
- **ملاحظة الأدوات:** لم يتم العثور على أدوات فحص ثغرات الاعتماديات (`pip-audit`, `safety`, `trivy`) مثبتة في البيئة، ولم يتم تثبيت أو تعديل أي مكتبات التزاماً بشروط التدقيق للقراءة فقط.

---

## 1. جرد الاعتماديات المعلنة والمثبتة

### الاعتماديات المعلنة (`pyproject.toml` و`requirements.txt`)
- `requires-python = ">=3.11"` (`pyproject.toml:11`)
- `torch>=2.2` (`pyproject.toml:13`, `requirements.txt:1`)
- `numpy>=1.26` (`pyproject.toml:14`, `requirements.txt:2`)
- `pytest>=8.0` (`pyproject.toml:19`, `requirements.txt:3`)

### الحزم المثبتة فعلياً في `.venv` (Python `3.15.2`)
- `torch 2.15.0`
- `numpy 2.5.4`
- `pytest 9.2.0`
- `Jinja2 3.1.6`
- `MarkupSafe 3.0.4`
- `setuptools 84.2.1`
- `sympy 1.14.0`, `networkx 3.7`, `fsspec 2026.9.0`, `filelock 4.2.10`, `mpmath 1.3.0`, `packaging 26.3`, `pluggy 1.7.0`, `Pygments 2.21.0`, `colorama 0.4.6`, `iniconfig 2.3.1`, `typing_extensions 4.16.0`, `pip 26.3`

---

## 2. نتائج تدقيق الاعتماديات

### `DEP-01` — غياب تثبيت الإصدارات (`Version Pinning`) وملف قفل (`Lockfile`)
- **الخطورة:** `Medium`
- **الملف والأسطر:** `pyproject.toml:12-20`, `requirements.txt:1-3`, `Dockerfile:4`, `render.yaml:6`
- **الوصف والدليل:** تستخدم جميع ملفات المتطلبات حدوداً دنيا مفتوحة (`torch>=2.2`, `numpy>=1.26`) دون تثبيت إصدارات محددة أو `requirements.lock` بالتحقق من الهاش (`--require-hashes`).
- **الأثر:** عدم قابلية إعادة بناء بيئة مطابقة في الإنتاج (Supply-Chain / Reproducibility Risk)، وإمكانية سحب إصدارات مستقبلية تكسر التوافق أو تحتوي على ثغرات غير مختبرة.
- **الحل العملي:** توليد ملف متطلبات مقفل بإصدارات دقيقة (`pip freeze` أو `uv pip compile --generate-hashes`) وتثبيت الإصدارات في بيئة الإنتاج (`Dockerfile` و`render.yaml`).

### `DEP-02` — تباين إصدار Python بين البيئة المحلية (`3.15.2`) وبيئة النشر (`3.11`)
- **الخطورة:** `Medium`
- **الملف والأسطر:** `Dockerfile:1` (`python:3.11-slim`), `render.yaml:10` (`3.11.9`), والبيئة المحلية (`.venv`: `Python 3.15.2`)
- **الوصف والدليل:** البيئة المحلية تعمل على Python 3.15.2 بينما النشر محدد بـ Python 3.11 / 3.11.9.
- **الأثر:** قد تمر اختبارات محلية بنجاح بينما تظهر اختلافات سلوكية أو أخطاء توافق في الإنتاج (أو العكس).
- **الحل العملي:** توحيد إصدار Python المستهدف في التطوير وCI والإنتاج (مثلاً Python 3.11 أو 3.12).

### `DEP-03` — الاعتماد على خدمات خارجية غير مثبتة العقود في وقت التشغيل
- **الخطورة:** `Medium`
- **الملف والأسطر:** `nourai/web_search.py:50, 102, 131, 310, 545, 687, 715`
- **الوصف والدليل:** يعتمد المحرك في وقت التشغيل على عدة واجهات خارجية (`wikipedia.org`, `news.google.com`, `api.duckduckgo.com`, `image.pollinations.ai`, `text.pollinations.ai`, `open.er-api.com`, `generativelanguage.googleapis.com`) بدون تثبيت إصدار أو تحقق من سلامة المحتوى المرجع.
- **الأثر:** توفر الخدمة وجودة المخرجات وخصوصية البيانات تعتمد على أطراف ثالثة قد تتغير واجهاتها أو تتعطل دون إنذار.
- **الحل العملي:** عزل هذه التكاملات خلف واجهات (`Adapters`) مع قواطع دائرة (`Circuit Breakers`) وإلغاء الاعتماد الصامت على نقاط النهاية غير الموثقة.

### `DEP-04` — عدم وجود فحص آلي للثغرات (`pip-audit` / `trivy`) في خط التكامل المستمر
- **الخطورة:** `Low`
- **الملف والأسطر:** `.github/workflows/` (يحتوي فقط على `firebase-hosting-merge.yml` ولا يحتوي على workflow لاختبار Python أو فحص الاعتماديات)
- **الحل العملي:** إضافة GitHub Actions workflow يشغّل `pytest` و`pip-audit` و`ruff` عند كل Pull Request.

---

## 3. تقييم الثغرات المعروفة (CVE Status)
- **الحزم الأساسية:** يعتمد المشروع على سطح اعتماديات صغير جداً في وقت التشغيل (`torch` و`numpy` والمكتبة القياسية لبايثون)، وهو أمر إيجابي يقلل سطح الهجوم.
- **حالة الفحص الآلي:** نظراً لعدم توفر `pip-audit` أو `trivy` أو الاتصال بقاعدة CVE داخل بيئة التدقيق الحالية، لم يتم تأكيد ثغرات CVE محددة في `torch 2.15.0` أو `numpy 2.5.4`، ويُوصى بتشغيل `pip-audit` في CI للتحقق الدوري.
