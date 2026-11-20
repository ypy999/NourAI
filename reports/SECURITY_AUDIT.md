# تقرير التدقيق الأمني الشامل — مشروع NourAI (`SECURITY_AUDIT.md`)

- **نطاق التدقيق:** فحص للقراءة فقط (Read-Only) لملفات الواجهة البرمجية (`api/`), واجهة المستخدم (`ui/`), النشر والإنتاج (`deployment/`, `run_nourai.py`, `Dockerfile`, `firebase.json`), محركات الذكاء الاصطناعي والبحث والذاكرة (`nourai/`, `inference/`, `model/`, `training/`).
- **تاريخ التدقيق:** 2026-11-17
- **القاعدة المرجعية:** جميع النتائج مدعومة بمسارات ملفات وأرقام أسطر فعلية تم التحقق منها مباشرة في الكود المصدري.

---

## ملخص النتائج حسب الخطورة

| مستوى الخطورة | العدد | المعرفات |
|---|---:|---|
| **Critical** | 2 | `SEC-01`, `SEC-02` |
| **High** | 7 | `SEC-03`, `SEC-04`, `SEC-05`, `SEC-06`, `SEC-07`, `SEC-08`, `SEC-09` |
| **Medium** | 6 | `SEC-10`, `SEC-11`, `SEC-12`, `SEC-13`, `SEC-14`, `SEC-15` |
| **Low** | 3 | `SEC-16`, `SEC-17`, `SEC-18` |

---

## 1. ثغرات حرجة (Critical)

### `SEC-01` — تصعيد الصلاحيات عبر التسجيل الذاتي غير المقيد (`Privilege Escalation`)
- **الخطورة:** `Critical`
- **الملف والأسطر:** `api/backend.py` (الأسطر `588-597`, `82-109`)
- **الوصف والدليل:**
  يقبل مسار التسجيل العام `POST /api/v1/users/register` حقل `role` مباشرة من جسم الطلب (`body.get("role", "user")`) دون أي تحقق أو تقييد:
  ```python
  user = self.users.register_user(
      str(body.get("username", "")),
      str(body.get("password", "")),
      role=str(body.get("role", "user")),
  )
  ```
- **الأثر:** يمكن لأي مهاجم غير مصادق عليه إنشاء حساب بصلاحية `"admin"` والحصول على رمز JWT صالح والوصول إلى مسارات الإدارة الحساسة مثل `GET /api/v1/admin/users`.
- **الحل العملي:**
  1. تثبيت `role="user"` قسراً في مسار `POST /api/v1/users/register` وتجاهل أي قيمة `role` قادمة من العميل.
  2. قصر إنشاء حسابات `admin` على أداة CLI محلية أو مسار إداري محمي بتوكن `admin` موجود مسبقاً، مع التحقق في `UserManager.register_user` من أن `role in ("user", "admin")`.

### `SEC-02` — تنفيذ أوامر عن بُعد وتجاوز العزل في `/api/v1/code/run` (`Remote Code Execution / Sandbox Bypass`)
- **الخطورة:** `Critical`
- **الملف والأسطر:** `api/backend.py` (الأسطر `683-703`)
- **الوصف والدليل:**
  ينفذ المسار `POST /api/v1/code/run` شيفرة Python مرسلة من أي مستخدم مصادق عليه (بما في ذلك مستخدمي الضيوف عبر `POST /api/v1/auth/guest`) عبر `subprocess.run([sys.executable, "-I", "-c", code_str], ...)` مع الاعتماد فقط على قائمة حظر نصية ساذجة:
  ```python
  forbidden = ("import subprocess", "import shutil", "import socket", "os.system", "os.remove", "os.unlink", "rmdir", "open(")
  if any(f in code_str for f in forbidden):
      return 400, ...
  ```
- **الأثر:** يمكن تجاوز قائمة الحظر بسهولة تامة عبر `__import__("os").popen(...)` أو `import pathlib` أو `from subprocess import Popen` أو `getattr` أو `eval()`/`exec()`. وبما أن العملية الفرعية تعمل بنفس صلاحيات المستخدم والملفات والشبكة الخاصة بالخادم، يمكن للمهاجم قراءة `data/production/.jwt_secret` وقاعدة البيانات `nourai_production.sqlite3`، وتنفيذ أوامر نظام كاملة، أو شن هجمات SSRF من داخل الخادم.
- **الحل العملي:**
  1. تعطيل `/api/v1/code/run` في بيئة الإنتاج أو قصره على وضع محلي معزول صراحةً (`NOURAI_ALLOW_LOCAL_CODE_EXEC=1` مع ربط `127.0.0.1` فقط).
  2. في حال الحاجة لتنفيذ الكود، يجب استخدام حاوية معزولة بدون شبكة (`--network=none`, `--read-only`, `--cap-drop=ALL`, حدود ذاكرة/CPU) أو تشغيله على جانب العميل عبر WebAssembly (Pyodide) داخل `iframe` معزول.

---

## 2. ثغرات عالية الخطورة (High)

### `SEC-03` — ثغرة XSS عبر حقن SVG خام في واجهة المستخدم (`Stored / Reflected XSS`)
- **الخطورة:** `High`
- **الملف والأسطر:** `ui/index.html` (الأسطر `1757`, `1761`)
- **الوصف والدليل:**
  عندما يحتوي رد المحادثة على كتلة ```svg``` تبدأ بـ `<svg`، يتم حقنها مباشرة داخل `targetEl.innerHTML` دون أي تعقيم (`DOMPurify` أو إزالة وسوم `<script>` ومعالجات الأحداث `onload`/`onerror`):
  ```javascript
  const inlineSvg = lang === "svg" && code.trim().startsWith("<svg") ? `<div style="margin:0.5rem 0;">${code}</div>` : "";
  ```
- **الأثر:** أي محتوى يحتوي على `<svg onload="...">` (سواء من استجابة LLM خارجية، أو نتيجة بحث ويب، أو مستند RAG، أو حقن موجه) ينفّذ JavaScript في أصل الصفحة، مما يمكّن من سرقة `nourai_token` و`nourai_gemini_key` من `document.cookie` / `localStorage`.
- **الحل العملي:**
  1. عدم إدراج SVG كـ HTML خام في DOM؛ بدلاً من ذلك عرضه داخل `<iframe sandbox>` أو كـ `data:image/svg+xml` أو تعقيمه بمكتبة مثل `DOMPurify` مع حظر `<script>`, `<foreignObject>`, وسمات `on*`.

### `SEC-04` — ثغرة DOM XSS عبر كسر علامة الاقتباس المفردة داخل `onclick`
- **الخطورة:** `High`
- **الملف والأسطر:** `ui/index.html` (الأسطر `1538`, `1568`, `1584`, `1755`, `1756`, `1758`)
- **الوصف والدليل:**
  تستخدم الواجهة `encodeURIComponent(...)` داخل سلسلة نصية محاطة بعلامتي اقتباس مفردتين `'...'` في سمات `onclick="..."`:
  ```javascript
  onclick="runPythonCode(decodeURIComponent('${encodedCode}'), '${outId}')"
  ```
  الدالة القياسية `encodeURIComponent` في JavaScript **لا تُرمّز علامة الاقتباس المفردة `'`**، وبالتالي أي `'` داخل الكود أو مخرجات JSON يكسر السلسلة النصية ويسمح بتنفيذ JavaScript.
- **الأثر:** تنفيذ شيفرة JavaScript عشوائية عند نقر المستخدم على أزرار التشغيل أو المعاينة أو النسخ أو الاختبار/المخطط.
- **الحل العملي:**
  استبدال معالجات `onclick` المضمنة بـ `data-*` attributes مع `addEventListener` (event delegation) و`textContent` / `dataset` بدلاً من تجميع نصوص HTML/JS ديناميكياً.

### `SEC-05` — ثغرة Server-Side Request Forgery (SSRF) في جلب الروابط
- **الخطورة:** `High`
- **الملف والأسطر:** `nourai/web_search.py` (الأسطر `318-349`)
- **الوصف والدليل:**
  تقوم الدالة `_maybe_read_and_summarize_url` باستخراج أي رابط `http://` أو `https://` من رسالة المستخدم وفتحه مباشرة من الخادم عبر `urlopen(req, timeout=5.5)` دون التحقق من عنوان IP المحلول أو النطاقات الداخلية، ودون حد أقصى لحجم الاستجابة المقروءة (`resp.read()`).
- **الأثر:** يمكن للمهاجم استغلال الخادم للوصول إلى خدمات محلية (`http://127.0.0.1:...`) أو بيانات Cloud Metadata (`http://169.254.169.254/latest/meta-data/`) أو فحص الشبكة الداخلية، أو التسبب في استهلاك الذاكرة (OOM) عبر استجابة ضخمة.
- **الحل العملي:**
  1. تحليل اسم النطاق إلى IP قبل الاتصال ورفض أي عناوين `is_private`, `is_loopback`, `is_link_local`, `is_reserved`, `is_multicast`.
  2. تعطيل إعادة التوجيه التلقائي أو التحقق من كل وجهة إعادة توجيه، وتحديد حد أقصى للقراءة (مثلاً `resp.read(1_000_000)`).

### `SEC-06` — تسريب بيانات المستخدم الحساسة ومستندات RAG إلى طرف ثالث خارجي (`Pollinations AI`)
- **الخطورة:** `High`
- **الملف والأسطر:** `nourai/web_search.py` (الأسطر `668-732`)
- **الوصف والدليل:**
  عند تفعيل `enable_web_search=True` وعدم توفر `GEMINI_API_KEY` (أو فشل الطلب)، ترسل الدالة `_synthesize_with_free_llm` تلقائياً: سؤال المستخدم، آخر 4 أدوار من المحادثة (`history[-4:]`)، **الذكريات المحفوظة للمستخدم (`user_memories`)**، و**مقاطع مستندات RAG الخاصة بالمستخدم (`rag_chunks`)** إلى نقطة النهاية الخارجية `https://text.pollinations.ai/openai` دون موافقة صريحة أو تنبيه.
- **الأثر:** انتهاك صريح لمبدأ الخصوصية المحلية (`Local-First`)؛ قد تُرسل بيانات شخصية أو مستندات سرية لمستخدمي النظام إلى خادم خارجي غير خاضع لسيطرة المشروع.
- **الحل العملي:**
  1. جعل إرسال البيانات إلى مزودين خارجيين (Pollinations / Gemini) خياراً صريحاً (`allow_cloud_synthesis=False` افتراضياً)، وعدم إرسال `user_memories` أو `rag_chunks` لمزود خارجي مجاني غير موثوق دون موافقة صريحة من المستخدم.

### `SEC-07` — غياب حد لحجم جسم الطلب (`Content-Length`) في خادم HTTP (DoS / OOM)
- **الخطورة:** `High`
- **الملف والأسطر:** `deployment/production.py` (الأسطر `251-253`)
- **الوصف والدليل:**
  يقرأ الخادم طول الطلب مباشرة دون حد أقصى ودون معالجة استثناءات التحويل:
  ```python
  length = int(self.headers.get("Content-Length", "0"))
  raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
  body = json.loads(raw_body) if raw_body.strip() else {}
  ```
- **الأثر:** إرسال `Content-Length` ضخم (مثل عدة غيغابايت) يؤدي إلى استهلاك الذاكرة وانهيار الخادم (Denial of Service). كما أن إرسال JSON غير صالح أو `Content-Length` غير رقمي يؤدي إلى استثناء غير معالج.
- **الحل العملي:**
  فرض حد أقصى لحجم الطلب (مثلاً `MAX_BODY_BYTES = 2 * 1024 * 1024`، أي 2 MiB، وإرجاع `413 Payload Too Large` عند تجاوزه)، ومعالجة `ValueError`, `UnicodeDecodeError`, و`json.JSONDecodeError` بإرجاع `400 Bad Request`.

### `SEC-08` — تجاهل عنوان IP الفعلي للعميل في `start_live_http_server`
- **الخطورة:** `High`
- **الملف والأسطر:** `deployment/production.py` (الأسطر `255-261`)، `api/backend.py` (السطر `503`)
- **الوصف والدليل:**
  يستدعي `_Handler._dispatch` الدالة `backend.handle_request(...)` دون تمرير `client_ip=self.client_address[0]`. وبما أن القيمة الافتراضية في `api/backend.py:503` هي `client_ip: str = "127.0.0.1"`، فإن **جميع العملاء عبر الشبكة يتشاركون نفس مفتاح تحديد المعدل (`guest:127.0.0.1`)**.
- **الأثر:** يمكن لمهاجم واحد استنفاد حصة `POST /api/v1/auth/guest` لجميع المستخدمين الآخرين (DoS)، كما تفقد سجلات/حدود معدل الطلبات دلالتها حسب العميل.
- **الحل العملي:**
  تمرير `client_ip=self.client_address[0]` من `_Handler._dispatch` إلى `backend.handle_request` (مع دعم ترويسة `X-Forwarded-For` فقط عندما يكون الخادم خلف Reverse Proxy موثوق).

### `SEC-09` — غياب `Rate Limiting` على تسجيل الدخول والتسجيل، ونمو غير محدود لذاكرة `RateLimiter`
- **الخطورة:** `High`
- **الملف والأسطر:** `api/backend.py` (الأسطر `588-609`, `189-210`)
- **الوصف والدليل:**
  المساران `POST /api/v1/auth/token` و`POST /api/v1/users/register` يقعان قبل فحص `self.rate_limiter.allow(...)` ولا يطبقان أي حد معدل طلبات (Rate Limit). بالإضافة إلى ذلك، قاموس `RateLimiter._events` لا يحذف المفاتيح المنتهية الصلاحية أبداً.
- **الأثر:** إمكانية تنفيذ هجمات التخمين (`Brute Force` / `Credential Stuffing`) على كلمات المرور، وإغراق قاعدة البيانات بحسابات وهمية، ونمو استهلاك الذاكرة تدريجياً في `RateLimiter._events`.
- **الحل العملي:**
  تطبيق `self.rate_limiter.allow(f"auth:{client_ip}")` على `/api/v1/auth/token` و`/api/v1/users/register`، وتنظيف المفاتيح الفارغة في `RateLimiter`.

---

## 3. ثغرات متوسطة الخطورة (Medium)

### `SEC-10` — غياب علامتي `HttpOnly` و`Secure` على كوكي المصادقة `nourai_token` وتضارب مدة الصلاحية
- **الخطورة:** `Medium`
- **الملف والأسطر:** `api/backend.py` (الأسطر `607`, `619`, `152`)
- **الوصف والدليل:**
  تُرسل الترويسة:
  ```python
  sec_headers["Set-Cookie"] = f"nourai_token={token}; Path=/; SameSite=Lax; Max-Age=604800"
  ```
  بدون `HttpOnly` (مما يعرضها للقراءة عبر `document.cookie` عند أي ثغرة XSS) وبدون `Secure` عند العمل عبر HTTPS. كما أن `Max-Age=604800` (7 أيام) بينما صلاحية التوكن في `AuthManager` هي `3600` ثانية (ساعة واحدة).
- **الحل العملي:** إضافة `HttpOnly`، وإضافة `Secure` عند تفعيل TLS/HTTPS، ومواءمة `Max-Age` مع `ttl_seconds`.

### `SEC-11` — تخزين مفتاح `Gemini API Key` في `document.cookie` وقراءته عبر JavaScript
- **الخطورة:** `Medium`
- **الملف والأسطر:** `ui/index.html` (الأسطر `1040`, `3046`)
- **الوصف والدليل:** يتم حفظ `nourai_gemini_key` في كوكي غير محمي بـ `HttpOnly`، مما يجعله عرضة للسرقة عند أي ثغرة XSS، كما يُمرّر لاحقاً في رابط URL (`?key=...`) في `nourai/web_search.py:687`.
- **الحل العملي:** تجنب تخزين مفاتيح API دائماً في الكوكيز؛ واستخدام `sessionStorage` كحد أقصى في الوضع العميل-فقط أو إدارتها من جهة الخادم عبر متغير البيئة `GEMINI_API_KEY`، وتمرير المفتاح عبر ترويسة `x-goog-api-key` بدلاً من Query String.

### `SEC-12` — تجاوز فحص أصل CORS (`Origin Prefix Bypass`) وغياب `Vary: Origin`
- **الخطورة:** `Medium`
- **الملف والأسطر:** `api/backend.py` (الأسطر `517-526`)
- **الوصف والدليل:**
  ```python
  allow_origin = origin if (origin in allowed_origins or origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost")) else "http://127.0.0.1"
  ```
  استخدام `startswith("http://localhost")` يسمح بنطاقات خبيثة مثل `http://localhost.attacker.com`!
- **الحل العملي:** استخدام `urllib.parse.urlparse(origin)` والتحقق من أن `parsed.hostname in ("127.0.0.1", "localhost")` وإضافة ترويسة `"Vary": "Origin"`.

### `SEC-13` — إمكانية تحويل مسار API عبر معامل الرابط `?api=` وتخزينه في `localStorage`
- **الخطورة:** `Medium`
- **الملف والأسطر:** `ui/index.html` (الأسطر `1107-1115`)
- **الوصف والدليل:** أي رابط بصيغة `?api=https://attacker.example` يحفظ عنوان الخادم المهاجم في `localStorage.getItem("nourai_api_base")` ويوجه جميع طلبات المحادثة والوثائق والذكريات إليه لاحقاً.
- **الحل العملي:** قصر `isSafeApiUrl` على قائمة بيضاء صريحة من النطاقات الموثوقة (بالإضافة إلى `localhost`/`127.0.0.1`) وعدم حفظ القيمة تلقائياً من URL بدون تأكيد المستخدم.

### `SEC-14` — خطر Zip Bomb / استنفاد الذاكرة عند فك ضغط ملفات `.docx`
- **الخطورة:** `Medium`
- **الملف والأسطر:** `nourai/memory_and_rag.py` (الأسطر `22-30`)
- **الوصف والدليل:** `zf.read("word/document.xml")` يقرأ الملف المضغوط بالكامل في الذاكرة دون التحقق من `zf.getinfo("word/document.xml").file_size` أو حجم `raw_bytes`.
- **الحل العملي:** التحقق من حجم `raw_bytes` (مثلاً <= 5 MiB) وحجم `file_size` غير المضغوط داخل الأرشيف قبل `zf.read(...)`.

### `SEC-15` — مخاطر أمن الذكاء الاصطناعي: حقن الموجهات غير المباشر (`Indirect Prompt Injection`) وغياب التصفية الدلالية
- **الخطورة:** `Medium`
- **الملف والأسطر:** `nourai/web_search.py` (الأسطر `665-680`), `api/backend.py` (الأسطر `439-446`, `765-766`), `nourai/capabilities.py` (الأسطر `19-37`, `91-102`)
- **الوصف والدليل:**
  1. يتم دمج نتائج الويب ومقاطع RAG وذكريات المستخدم مباشرة في نص `user_payload` دون فواصل عازلة أو تعليمات تجاهل الأوامر المضمنة في البيانات الخارجية.
  2. فحص الأمان في `SAFETY_BLOCKLIST_PATTERNS` و`evaluate_prompt_safety` يعتمد على مطابقة سلاسل نصية ثابتة بالـ `lowercase` فقط، مما يسهل تجاوزه بترميز أو صياغة بديلة.
  3. في `nourai/engines.py:39-49` دالة `_safe_eval_ast` تسمح بعملية الأس `ast.Pow` دون حد أقصى للأس (`2 ** 100000000`) مما قد يسبب استهلاك CPU/ذاكرة.
- **الحل العملي:** عزل المحتوى الخارجي داخل وسوم واضحة مع توجيه النظام لعدم تنفيذ تعليمات من السياق المسترجع، وتقييد قيمة الأس في `_safe_eval_ast` (مثلاً `abs(right) <= 1000`).

---

## 4. ثغرات منخفضة الخطورة (Low)

### `SEC-16` — بقاء بيانات RAG بعد حذف حساب المستخدم (`DELETE /api/v1/users/me`)
- **الخطورة:** `Low`
- **الملف والأسطر:** `api/backend.py` (الأسطر `633-641`, `93`)
- **الوصف والدليل:** عند حذف الحساب يتم حذف الجلسات والذكريات، لكن لا يتم حذف مستندات RAG عبر `self.rag`. وبما أن `user_id` حتمي من اسم المستخدم (`sha256(username)[:12]`)، فإن إعادة تسجيل نفس اسم المستخدم تستعيد الوصول لمستندات RAG القديمة.
- **الحل العملي:** حذف مستندات RAG الخاصة بالمستخدم عند حذف الحساب، واستخدام `user_id` عشوائي (`secrets.token_hex(8)`) بدلاً من اشتقاقه حتمياً من اسم المستخدم.

### `SEC-17` — صلاحيات الملفات الحساسة (`0644` / `chmod -R 755 /app`) وإرسال HSTS فوق HTTP غير المشفر
- **الخطورة:** `Low`
- **الملف والأسطر:** `run_nourai.py` (الأسطر `116-120`), `deployment/production.py` (الأسطر `188`, `262`), `Dockerfile` (السطر `6`)
- **الوصف والدليل:** يتم إنشاء `.jwt_secret` و`localhost_key.pem` بصلاحيات افتراضية مقروءة للجميع على أنظمة POSIX، كما يضيف الخادم ترويسة `Strict-Transport-Security` حتى عند تشغيله بدون TLS.
- **الحل العملي:** تعيين صلاحيات `0600` (`os.chmod(..., 0o600)`) لملفات المفاتيح والأسرار، وإرسال HSTS فقط عندما يكون الاتصال عبر HTTPS (`enable_tls=True`).

### `SEC-18` — خيار احتياطي قديم في `torch.load` بدون `weights_only=True`
- **الخطورة:** `Low`
- **الملف والأسطر:** `model/transformer.py` (الأسطر `299-303`)
- **الوصف والدليل:** في حال حدوث `TypeError` يسقط الكود إلى `torch.load(path, map_location=map_location)` بدون `weights_only=True` (مما قد يسمح بتنفيذ كود عبر `pickle` إذا تم تحميل نقطة تفتيش غير موثوقة على إصدار PyTorch قديم).
- **الحل العملي:** اشتراط `PyTorch >= 2.2` (المحدد أصلاً في `pyproject.toml`) وإزالة المسار الاحتياطي غير الآمن.
