# تقرير التدريب العصبي الموسّع ومقارنة التوليد قبل/بعد (`v1.2.0`)

- **الحالة الهندسية:** `ENGINEERING_INFRA_VERIFIED_NANO_843K_MODEL`
- **عدد المعاملات:** `843,008` (`Byte-Level Pre-LN Transformer`, `context_length=128`)
- **خطوات التدريب:** `150` خطوة Pre-train + `400` خطوة SFT (مقنّعة الخسارة على الردود)
- **بصمة النموذج المحدث (`SHA-256`):** `6228c42a78a26625c50c5bc2a66fddd6c3f4d43e6ab1a34daf83fc1df0a95409`

## مقارنة التوليد العصبي المباشر (`InferenceEngine.generate`) قبل وبعد التدريب الموسّع

| اللغة | المدخل | التوليد قبل (60+30 خطوة) | التوليد بعد (`<|user|>...<|assistant|>`) | التوليد بعد (نص مباشر) |
|:---:|---|---|---|---|
| `ar` | `مرحبا، كيف حالك؟` | ` الايالامالالالالالال�` | `de للحموطاي ماانعبوالان` | ` الامري لتاواهبة الااا�` |
| `ar` | `ما هي عاصمة قطر؟` | ` الايامالالالالالالال�` | `Tra + (لاة 1`pustengat). اواالال` | ` يالاااهاا لاوالأانب م�` |
| `en` | `Hello, how are you?` | ` dentas de de le de dente de de dentente d` | `�calile alist alist aligmolile dis a al al` | ` lintes das de le alilis. dete last latis ` |
| `en` | `What is the capital of France?` | ` de de de dentas de de de de de de le de d` | `` | ` Lal dentis caral den de de lent le lou l ` |
| `fr` | `Bonjour, comment allez-vous ?` | ` dentas de de de de le de de de de dente d` | `Les a de pite ome leng e de ontis lirintan` | ` les dint l le ont le le le les d les le p` |
| `fr` | `Quelle est la capitale de la France ?` | ` de dentas de de de de de de le de de de d` | `Te ontis Lal des Lal lis Les pis.` | ` des de li oue la de re de le le de en e d` |
