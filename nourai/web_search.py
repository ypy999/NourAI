"""Zero-cost, zero-bloat Live Web Search & Universal Q&A Engine for NourAI (Wikipedia + Google News RSS + DuckDuckGo + Free LLM Synthesis)."""
from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from nourai.capabilities import NourAICapabilityEngine, detect_text_language, evaluate_prompt_safety


USER_AGENT = "NourAI/1.2.0 (https://github.com/ypy999/NourAI; Local-First Multilingual Assistant)"


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_search_query(query: str) -> str:
    """Extract the core search topic from conversational Arabic, English, or French questions."""
    q = query.strip()
    prefixes = [
        r"^(ابحث في قوقل عن|ابحث عن|ابحث لي عن|ما هو|ما هي|من هو|من هي|أخبرني عن|معلومات عن|كم|متى|أين)\s+",
        r"^(search google for|search for|what is|what are|who is|who was|tell me about|where is|when was)\s+",
        r"^(cherche sur google|recherche|qu'est-ce que|qu'est ce que|qui est|parle-moi de|où est)\s+",
    ]
    for pat in prefixes:
        q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()
    q = q.rstrip("؟?!.")
    return q or query.strip()


def search_wikipedia(query: str, lang_code: str = "ar", max_results: int = 3) -> list[dict[str, str]]:
    """Query Wikipedia MediaWiki API (ar/en/fr) for instant encyclopedic summaries and URLs ($0, no API key)."""
    clean_q = _clean_search_query(query)
    wiki_lang = lang_code if lang_code in ("ar", "en", "fr") else "ar"
    results: list[dict[str, str]] = []

    for lg in ([wiki_lang, "en"] if wiki_lang != "en" else ["en", "ar"]):
        if results:
            break
        api_url = (
            f"https://{lg}.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={quote_plus(clean_q)}&utf8=1&format=json&srlimit={max_results}"
        )
        try:
            req = Request(api_url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=4.5) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            search_items = data.get("query", {}).get("search", [])
            for item in search_items[:max_results]:
                title = item.get("title", "")
                snippet = _strip_html(item.get("snippet", ""))
                page_url = f"https://{lg}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

                # Fetch fuller intro extract for the top match
                if not results and title:
                    summary_url = (
                        f"https://{lg}.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title.replace(' ', '_'))}"
                    )
                    try:
                        s_req = Request(summary_url, headers={"User-Agent": USER_AGENT})
                        with urlopen(s_req, timeout=3.5) as s_resp:
                            s_data = json.loads(s_resp.read().decode("utf-8", errors="replace"))
                            extract = str(s_data.get("extract") or "").strip()
                            if extract:
                                snippet = extract
                    except Exception:
                        pass

                if title and snippet:
                    results.append(
                        {
                            "source": f"Wikipedia ({lg.upper()})",
                            "title": title,
                            "snippet": snippet,
                            "url": page_url,
                        }
                    )
        except Exception:
            continue
    return results


def search_google_news_rss(query: str, lang_code: str = "ar", max_results: int = 3) -> list[dict[str, str]]:
    """Query Google News RSS feed for live web/news results ($0, no API key)."""
    clean_q = _clean_search_query(query)
    if lang_code == "ar":
        hl, gl, ceid = "ar", "SA", "SA:ar"
    elif lang_code == "fr":
        hl, gl, ceid = "fr", "FR", "FR:fr"
    else:
        hl, gl, ceid = "en-US", "US", "US:en"

    rss_url = f"https://news.google.com/rss/search?q={quote_plus(clean_q)}&hl={hl}&gl={gl}&ceid={ceid}"
    results: list[dict[str, str]] = []
    try:
        req = Request(rss_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=4.5) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            if title:
                results.append(
                    {
                        "source": "Google Search / News",
                        "title": title,
                        "snippet": desc if desc and desc != title else f"تاريخ النشر: {pub_date}",
                        "url": link,
                    }
                )
    except Exception:
        pass
    return results


def search_duckduckgo_instant(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Query DuckDuckGo Instant Answer JSON API ($0, no API key)."""
    clean_q = _clean_search_query(query)
    url = f"https://api.duckduckgo.com/?q={quote_plus(clean_q)}&format=json&no_html=1&skip_disambig=1"
    results: list[dict[str, str]] = []
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        abstract = str(data.get("AbstractText") or "").strip()
        abstract_url = str(data.get("AbstractURL") or "").strip()
        heading = str(data.get("Heading") or clean_q).strip()
        if abstract:
            results.append(
                {
                    "source": "DuckDuckGo Instant",
                    "title": heading,
                    "snippet": abstract,
                    "url": abstract_url,
                }
            )
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "source": "DuckDuckGo Web",
                        "title": heading,
                        "snippet": str(topic["Text"]).strip(),
                        "url": str(topic.get("FirstURL") or ""),
                    }
                )
    except Exception:
        pass
    return results[:max_results]


def _answer_conversational_or_identity(prompt: str, lang_code: str, newly_saved_memories: list[str] | None = None) -> str | None:
    """Handle direct conversational, greeting, identity, and capability questions without searching Google."""
    clean = re.sub(r"[؟?!.,،]+", " ", (prompt or "").strip().lower())
    clean = re.sub(r"\s+", " ", clean).strip()

    # Normalize common Arabic spelling variants (أ/إ/آ -> ا, ة -> ه, ذ/د in مادا)
    norm_ar = (
        clean.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("مادا", "ماذا")
    )

    capability_triggers = (
        "ماذا يمكنك",
        "ماذا تستطيع",
        "ما هي قدراتك",
        "ماهي قدراتك",
        "ما الذي يمكنك",
        "كيف تساعدني",
        "كيف يمكنك مساعدتي",
        "وش تقدر",
        "شو بتعمل",
        "ماذا تفعل",
        "ايش تسوي",
        "what can you do",
        "what are your capabilities",
        "how can you help",
        "que peux-tu faire",
        "que pouvez-vous faire",
    )
    if any(t in norm_ar or t in clean for t in capability_triggers):
        if lang_code == "ar":
            return (
                "أنا **NourAI**، مساعدك الذكي متعدد اللغات (العربية، الإنجليزية، والفرنسية). إليك ما يمكنني فعله لك:\n\n"
                "1. **الإجابة عن الأسئلة والبحث الحي في الويب وقوقل:** اسألني عن أي شخصية، دولة، علم، أو خبر حديث.\n"
                "2. **البرمجة والمعاينة الحية (Live Code Canvas):** أكتب أكواد Python وHTML/JS مع زر **▶️ معاينة** لتشغيلها مباشرة.\n"
                "3. **تحليل الملفات المرفقة (📎 Local RAG):** ارفع أي ملف نصي أو برمجي واسألني عما بداخله.\n"
                "4. **الترجمة والتلخيص والرياضيات:** أترجم بين العربية والإنجليزية والفرنسية، وألخص النصوص، وأحل المسائل الرياضية.\n"
                "5. **الذاكرة الدائمة (🧠 Memory):** أتذكر اسمك وتفضيلاتك عبر جميع المحادثات."
            )
        if lang_code == "fr":
            return (
                "Je suis **NourAI**, votre assistant IA multilingue (arabe, anglais, français). Je peux :\n"
                "1. Répondre à vos questions et chercher en direct sur le Web / Wikipédia.\n"
                "2. Écrire du code Python/HTML avec aperçu en direct (▶️).\n"
                "3. Analyser vos fichiers joints (📎), traduire, résumer et calculer."
            )
        return (
            "I am **NourAI**, your multilingual AI assistant (Arabic, English, French). Here is what I can do:\n"
            "1. **Live Web & Wikipedia Q&A:** Answer general questions with live sources.\n"
            "2. **Coding & Live HTML Preview (▶️):** Write Python and interactive HTML/JS widgets.\n"
            "3. **Document Analysis (📎), Translation, Summarization, Math & Persistent Memory (🧠).**"
        )

    identity_triggers = (
        "من انت",
        "ما هو nourai",
        "ماهو nourai",
        "عرف عن نفسك",
        "عرفني بنفسك",
        "ما اسمك",
        "شو اسمك",
        "who are you",
        "what is nourai",
        "what is your name",
        "qui es-tu",
        "qu'est-ce que nourai",
    )
    if any(t in norm_ar or t in clean for t in identity_triggers):
        if lang_code == "ar":
            return "أنا **NourAI (`v1.2.0`)**، نموذج ومساعد ذكاء اصطناعي متعدد اللغات يعمل باللغات العربية والإنجليزية والفرنسية، ومزوّد بشبكة عصبية (`4.87M` معامل) وبحث ويب حي وذاكرة دائمة."
        if lang_code == "fr":
            return "Je suis **NourAI (`v1.2.0`)**, un modèle et assistant IA multilingue prenant en charge l'arabe, l'anglais et le français."
        return "I am **NourAI (`v1.2.0`)**, a multilingual AI model and assistant supporting Arabic, English, and French."

    greeting_triggers = (
        "مرحبا",
        "اهلا",
        "السلام عليكم",
        "صباح الخير",
        "مساء الخير",
        "كيف حالك",
        "كيفك",
        "شلونك",
        "اخبارك",
        "hello",
        "hi",
        "hey",
        "good morning",
        "how are you",
        "bonjour",
        "salut",
        "comment allez-vous",
        "comment ca va",
        "comment ça va",
    )
    if clean in greeting_triggers or norm_ar in greeting_triggers or any(
        norm_ar.startswith(g) and len(norm_ar.split()) <= 5 for g in greeting_triggers
    ):
        if lang_code == "ar":
            return "أهلاً بك في **NourAI**! أنا بخير وجاهز لمساعدتك. كيف يمكنني خدمتك اليوم؟"
        if lang_code == "fr":
            return "Bonjour ! Je suis **NourAI**, je vais très bien. Comment puis-je vous aider aujourd'hui ?"
        return "Hello! Welcome to **NourAI**. I am doing great—how can I help you today?"

    thanks_triggers = ("شكرا", "شكراً", "جزاك الله خيرا", "ممتاز", "رائع", "thank you", "thanks", "merci")
    if clean in thanks_triggers or norm_ar in thanks_triggers:
        if lang_code == "ar":
            return "على الرحب والسعة! يسعدني دائماً مساعدتك."
        if lang_code == "fr":
            return "Je vous en prie ! N'hésitez pas si vous avez d'autres questions."
        return "You're very welcome! Let me know if you need anything else."

    if any(norm_ar.startswith(p) for p in ("تذكر ان", "لا تنس ان", "احفظ في ذاكرتك", "اسمي ", "انا اسمي ")):
        if lang_code == "ar":
            return "تم حفظ هذه المعلومة في ذاكرتي الدائمة 🧠، وسأتذكرها في محادثاتنا القادمة."

    return None


def _maybe_generate_ai_image(prompt: str, lang_code: str) -> str | None:
    """Feature 1: Generate a high-res AI image URL ($0, no API key) when user asks to draw/create an image."""
    text = (prompt or "").strip()
    lower = text.lower()
    img_patterns = [
        r"^(?:ارسم|إرسم|صمم صورة|اصنع صورة|ولد صورة|تخيل صورة|ارسم لي صورة|ارسم لي)\s+(.+)$",
        r"^(?:draw|generate an image of|create an image of|create a picture of|imagine)\s+(.+)$",
        r"^(?:dessine|génère une image de|crée une image de)\s+(.+)$",
    ]
    subj = None
    for pat in img_patterns:
        m = re.match(pat, text, flags=re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            break
    if not subj and any(
        k in lower
        for k in ("ارسم صورة", "ولد صورة", "اصنع صورة", "generate image", "draw a picture", "générer une image")
    ):
        subj = text
    if not subj:
        return None

    seed = sum(ord(c) for c in subj) % 10000
    img_url = f"https://image.pollinations.ai/prompt/{quote_plus(subj)}?width=768&height=512&nologo=true&seed={seed}"
    if lang_code == "ar":
        return f"🎨 **تم توليد الصورة بالذكاء الاصطناعي لـ «{subj}»:**\n\n![{subj}]({img_url})\n\n[📥 فتح أو تنزيل الصورة بالحجم الكامل]({img_url})"
    if lang_code == "fr":
        return f"🎨 **Image générée par IA pour « {subj} » :**\n\n![{subj}]({img_url})\n\n[📥 Ouvrir / Télécharger l'image]({img_url})"
    return f"🎨 **AI Generated Image for '{subj}':**\n\n![{subj}]({img_url})\n\n[📥 Open / Download Full Image]({img_url})"


def _maybe_read_and_summarize_url(prompt: str, lang_code: str) -> str | None:
    """Feature 2: Fetch and summarize any http:// or https:// webpage URL pasted in the chat ($0)."""
    url_match = re.search(r"(https?://[^\s<>\"']+)", prompt or "")
    if not url_match:
        return None
    target_url = url_match.group(1).rstrip(".,;)")
    try:
        req = Request(target_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=5.5) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
        title_m = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw_html, flags=re.IGNORECASE)
        page_title = _strip_html(title_m.group(1)) if title_m else target_url
        cleaned = re.sub(r"<(script|style|noscript|svg|header|footer|nav)[^>]*>[\s\S]*?</\1>", " ", raw_html, flags=re.IGNORECASE)
        paragraphs = [
            _strip_html(p)
            for p in re.findall(r"<p[^>]*>([\s\S]*?)</p>", cleaned, flags=re.IGNORECASE)
        ]
        meaningful = [p for p in paragraphs if len(p) >= 45][:5]
        if not meaningful:
            body_txt = _strip_html(cleaned)
            meaningful = [body_txt[:600]] if body_txt else ["لم يتم العثور على نص مقروء كافٍ داخل الصفحة."]

        bullets = "\n".join(f"- {p[:280]}" for p in meaningful[:4])
        if lang_code == "ar":
            return f"🔗 **ملخص محتوى الرابط:** [{page_title}]({target_url})\n\n**أبرز النقاط المستخلصة من الصفحة:**\n{bullets}"
        if lang_code == "fr":
            return f"🔗 **Résumé de la page :** [{page_title}]({target_url})\n\n**Points clés extraits :**\n{bullets}"
        return f"🔗 **Webpage Summary:** [{page_title}]({target_url})\n\n**Key Extracted Points:**\n{bullets}"
    except Exception as exc:
        if lang_code == "ar":
            return f"⚠️ تعذّر الوصول المباشر إلى الرابط ({target_url}): `{exc}`"
        return f"⚠️ Could not fetch URL ({target_url}): `{exc}`"


def _maybe_proofread_and_rewrite(prompt: str, lang_code: str) -> str | None:
    """Feature 4: Smart Proofreader & Professional Rewriter for Arabic, English, and French."""
    text = (prompt or "").strip()
    patterns = [
        r"^(?:صحح|دقق|دقق لغويا|صحح هذا النص|أعد صياغة|اعد صياغة|حسّن صياغة)\s*[:：-]?\s*(.+)$",
        r"^(?:proofread|correct this|rewrite professionally|improve this text)\s*[:：-]?\s*(.+)$",
        r"^(?:corrige|corrigez|reformule|reformulez)\s*[:：-]?\s*(.+)$",
    ]
    target = None
    for pat in patterns:
        m = re.match(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            target = m.group(1).strip()
            break
    if not target:
        return None

    # Common Arabic typo & grammar fixes
    fixed = target
    ar_fixes = {
        "مادا": "ماذا",
        "هاذا": "هذا",
        "هاذه": "هذه",
        "لاكن": "لكن",
        "ذالك": "ذلك",
        "انشاء الله": "إن شاء الله",
        "ان شاء الله": "إن شاء الله",
        "مساءالخير": "مساء الخير",
        "صباحالخير": "صباح الخير",
        "شكراً جزيلا": "شكراً جزيلاً",
        "شكرا جزيلا": "شكراً جزيلاً",
        "انت ": "أنت ",
        "انا ": "أنا ",
        "الى ": "إلى ",
    }
    for wrong, right in ar_fixes.items():
        fixed = fixed.replace(wrong, right)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    if fixed and fixed[-1] not in ".!؟?":
        fixed += "."

    if lang_code == "ar":
        return (
            f"✍️ **التدقيق اللغوي وإعادة الصياغة الاحترافية:**\n\n"
            f"- **النص بعد التصحيح الإملائي والنحوي:**\n  «{fixed}»\n\n"
            f"- **الصياغة الرسمية المقترحة:**\n  «نودّ إفادتكم بأنّ: {fixed}»"
        )
    if lang_code == "fr":
        return f"✍️ **Correction et reformulation professionnelle :**\n\n- **Texte corrigé :** « {fixed[0].upper() + fixed[1:] if fixed else fixed} »"
    return f"✍️ **Proofreading & Professional Rewrite:**\n\n- **Corrected Text:** \"{fixed[0].upper() + fixed[1:] if fixed else fixed}\""


def _maybe_generate_html_artifact(prompt: str, lang_code: str) -> str | None:
    """Generate a complete runnable HTML/CSS/JS artifact when the user requests a web widget, calculator, or page."""
    lower = (prompt or "").lower()
    html_triggers = (
        "html",
        "آلة حاسبة",
        "الة حاسبة",
        "صفحة هبوط",
        "صفحة ويب",
        "لوحة تحكم",
        "calculator",
        "landing page",
        "calculatrice",
    )
    if not any(t in lower for t in html_triggers):
        return None

    intro = (
        "إليك كود تطبيق ويب تفاعلي كامل بـ **HTML/CSS/JS** جاهز للتجربة. اضغط على زر **«▶️ معاينة حية (Live Preview)»** أعلى صندوق الكود لتشغيله مباشرة داخل النافذة:"
        if lang_code == "ar"
        else (
            "Voici une application Web interactive complète en **HTML/CSS/JS**. Cliquez sur **« ▶️ Live Preview »** pour l'exécuter :"
            if lang_code == "fr"
            else "Here is a complete interactive **HTML/CSS/JS** web widget. Click **'▶️ Live Preview'** above the code block to run it live:"
        )
    )
    html_code = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <style>
    body { font-family: system-ui, sans-serif; background: linear-gradient(135deg,#0f172a,#1e1b4b); color: #fff; display: flex; justify-content: center; align-items: center; min-height: 92vh; margin: 0; }
    .card { background: rgba(255,255,255,0.08); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.18); padding: 22px; border-radius: 18px; width: 310px; box-shadow: 0 20px 40px rgba(0,0,0,0.45); }
    h3 { margin: 0 0 12px 0; text-align: center; background: linear-gradient(90deg,#60a5fa,#ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    #display { width: 100%; height: 48px; border-radius: 10px; border: none; background: #090d16; color: #38bdf8; font-size: 1.4rem; text-align: left; padding: 0 12px; margin-bottom: 12px; box-sizing: border-box; direction: ltr; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; direction: ltr; }
    button { height: 44px; border-radius: 10px; border: none; font-size: 1.05rem; font-weight: bold; cursor: pointer; background: #1e293b; color: #f8fafc; transition: 0.15s; }
    button:hover { background: #4f46e5; }
    .eq { background: linear-gradient(135deg,#6366f1,#ec4899); }
  </style>
</head>
<body>
  <div class="card">
    <h3>✦ حاسبة NourAI الذكية</h3>
    <input id="display" readonly value="0" />
    <div class="grid">
      <button onclick="clr()">C</button><button onclick="ins('(')">(</button><button onclick="ins(')')">)</button><button onclick="ins('/')">÷</button>
      <button onclick="ins('7')">7</button><button onclick="ins('8')">8</button><button onclick="ins('9')">9</button><button onclick="ins('*')">×</button>
      <button onclick="ins('4')">4</button><button onclick="ins('5')">5</button><button onclick="ins('6')">6</button><button onclick="ins('-')">−</button>
      <button onclick="ins('1')">1</button><button onclick="ins('2')">2</button><button onclick="ins('3')">3</button><button onclick="ins('+')">+</button>
      <button onclick="ins('0')" style="grid-column:span 2">0</button><button onclick="ins('.')">.</button><button class="eq" onclick="calc()">=</button>
    </div>
  </div>
  <script>
    const d = document.getElementById('display');
    function ins(v) { d.value = (d.value === '0' ? v : d.value + v); }
    function clr() { d.value = '0'; }
    function calc() { try { d.value = String(Function('"use strict";return (' + d.value + ')')()); } catch { d.value = 'Error'; } }
  </script>
</body>
</html>"""
    return f"{intro}\n\n```html\n{html_code}\n```"


def query_free_llm_with_context(
    prompt: str,
    *,
    lang_code: str,
    web_results: list[dict[str, str]],
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    user_memories: list[str] | None = None,
    rag_chunks: list[dict[str, Any]] | None = None,
) -> str | None:
    """Synthesize a ChatGPT/Gemini-grade answer using optional Gemini/Groq API key or free Pollinations AI endpoint."""
    context_lines = []
    for idx, r in enumerate(web_results[:5], 1):
        context_lines.append(f"[Web {idx}] {r['source']} - {r['title']}: {r['snippet']} ({r['url']})")
    if rag_chunks:
        for ch in rag_chunks[:3]:
            context_lines.append(f"[Document {ch['filename']} #{ch['chunk_index']}]: {ch['content']}")
    if user_memories:
        context_lines.append("User Saved Memories: " + " | ".join(user_memories))
    web_context = "\n".join(context_lines) if context_lines else "No external web snippets needed."

    sys_instruction = (
        "You are NourAI v1.2.0, an intelligent, helpful multilingual AI assistant (like ChatGPT and Google Gemini) "
        "supporting Arabic, English, and French. Answer accurately, clearly, and politely in the user's language. "
        "If live web search context is provided below, use it to ground your answer and cite the sources."
    )
    user_payload = f"User Question: {prompt}\n\nLive Web Search Results:\n{web_context}"

    # 1. Try Google Gemini Official API if GEMINI_API_KEY is provided
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            g_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            )
            body = {
                "contents": [{"parts": [{"text": f"{sys_instruction}\n\n{user_payload}"}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 700},
            }
            req = Request(
                g_url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
                method="POST",
            )
            with urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                return text
        except Exception:
            pass

    # 2. Try zero-key free Pollinations OpenAI-compatible text endpoint ($0, no API key required)
    try:
        messages = [{"role": "system", "content": sys_instruction}]
        for turn in (history or [])[-4:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_payload})
        req = Request(
            "https://text.pollinations.ai/openai",
            data=json.dumps({"model": "openai", "messages": messages, "seed": 42}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        with urlopen(req, timeout=8.5) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
        if raw.startswith("{"):
            parsed = json.loads(raw)
            choice_txt = (
                parsed.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            if choice_txt:
                return choice_txt
        elif len(raw) > 10:
            return raw
    except Exception:
        pass

    return None


class NourAIWebAndSmartEngine:
    """Unified zero-cost engine combining Local Capabilities, Live Google/Wikipedia Search, and Free AI Synthesis."""

    def __init__(self, capability_engine: NourAICapabilityEngine | None = None):
        self.capabilities = capability_engine or NourAICapabilityEngine()

    def answer_with_web(
        self,
        prompt: str,
        *,
        language: str | None = None,
        history: list[dict[str, str]] | None = None,
        enable_web_search: bool = True,
        gemini_api_key: str | None = None,
        user_memories: list[str] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
        persona: str = "general",
    ) -> dict[str, Any]:
        detected = detect_text_language(prompt)
        lang_code = "ar" if detected == "arabic" else ("fr" if detected == "french" else "en")
        if language:
            norm = language.lower().strip()
            if norm in ("ar", "arabic"):
                lang_code = "ar"
            elif norm in ("fr", "french"):
                lang_code = "fr"
            elif norm in ("en", "english"):
                lang_code = "en"

        # 1. Always enforce safety & prompt injection guard first
        safety = evaluate_prompt_safety(prompt)
        if not safety.get("is_safe", True):
            local_res = self.capabilities.answer(prompt, language=lang_code, history=history)
            return {
                "reply": local_res["reply"],
                "language": lang_code,
                "task": "safety_refusal",
                "response_source": "safety_guard",
                "citations": [],
            }

        # 1a. Check if user is asking for Free AI Image Generation (Feature 1)
        img_reply = _maybe_generate_ai_image(prompt, lang_code)
        if img_reply:
            return {
                "reply": img_reply,
                "language": lang_code,
                "task": "ai_image_generation",
                "response_source": "ai_image_generator",
                "citations": [],
            }

        # 1b. Check if user pasted a webpage URL to read & summarize (Feature 2)
        url_reply = _maybe_read_and_summarize_url(prompt, lang_code)
        if url_reply:
            return {
                "reply": url_reply,
                "language": lang_code,
                "task": "webpage_url_summary",
                "response_source": "url_reader",
                "citations": [],
            }

        # 1c. Check if user asked for Proofreading / Professional Rewriting (Feature 4)
        proof_reply = _maybe_proofread_and_rewrite(prompt, lang_code)
        if proof_reply:
            return {
                "reply": proof_reply,
                "language": lang_code,
                "task": "proofreading_and_rewrite",
                "response_source": "smart_proofreader",
                "citations": [],
            }

        # 1d. Check if user is asking for an interactive HTML/JS Preview Artifact (Canvas)
        html_artifact = _maybe_generate_html_artifact(prompt, lang_code)
        if html_artifact:
            return {
                "reply": html_artifact,
                "language": lang_code,
                "task": "html_live_artifact",
                "response_source": "live_code_canvas",
                "citations": [],
            }

        # 1c. Check if user is asking about their persistent Long-Term Memories
        lower_p = prompt.lower()
        if user_memories and any(
            k in lower_p
            for k in ("ما اسمي", "من أنا", "ماذا تعرف عني", "ما في ذاكرتك", "تذكرني", "what is my name", "what do you remember", "comment je m'appelle")
        ):
            mem_bullet = "\n".join(f"- 🧠 {m}" for m in user_memories)
            reply_mem = (
                f"بناءً على **الذاكرة الدائمة (Long-Term Memory)** المحفوظة لك عبر الجلسات:\n\n{mem_bullet}"
                if lang_code == "ar"
                else f"Based on your saved **Long-Term Memory** across sessions:\n\n{mem_bullet}"
            )
            return {
                "reply": reply_mem,
                "language": lang_code,
                "task": "long_term_memory_recall",
                "response_source": "long_term_memory",
                "citations": [],
            }

        # 1d. Check if user uploaded documents (Local Document RAG) and is asking about them
        if rag_chunks and any(
            k in lower_p
            for k in ("ملف", "مستند", "وثيقة", "المرفق", "في النص المرفوع", "file", "document", "uploaded", "fichier", "document")
        ):
            lines = [
                "📄 **الإجابة المستخلصة من ملفاتك المرفقة (Local Document RAG):**\n"
                if lang_code == "ar"
                else "📄 **Answer extracted from your uploaded documents (Local Document RAG):**\n"
            ]
            for ch in rag_chunks[:3]:
                lines.append(f"- **[ملف: `{ch['filename']}` — فقرة #{ch['chunk_index']}]**:\n  {ch['content']}")
            return {
                "reply": "\n\n".join(lines),
                "language": lang_code,
                "task": "document_rag",
                "response_source": "local_document_rag",
                "citations": [{"source": "Uploaded Document", "title": c["filename"], "snippet": c["content"][:120], "url": ""} for c in rag_chunks[:3]],
            }

        # 1e. Check if it's a direct conversational / greeting / identity / capability question
        conv_reply = _answer_conversational_or_identity(prompt, lang_code)
        if conv_reply:
            return {
                "reply": conv_reply,
                "language": lang_code,
                "task": "conversation_and_capabilities",
                "response_source": "nourai_assistant",
                "citations": [],
            }

        # 2. If it's a deterministic local task (math, coding, translation, memory, contradiction), check local capability first
        local_res = self.capabilities.answer(prompt, language=lang_code, history=history)
        if local_res["task"] in ("mathematics", "coding", "translation", "multi_turn_memory", "contradiction_detection", "summarization"):
            if not enable_web_search or "ابحث" not in prompt and "search" not in prompt.lower():
                return {
                    "reply": local_res["reply"],
                    "language": lang_code,
                    "task": local_res["task"],
                    "response_source": "smart_local_engine",
                    "citations": [],
                }

        # 3. Perform Live Multi-Source Web Search (Wikipedia + DuckDuckGo + Google News RSS)
        web_results: list[dict[str, str]] = []
        if enable_web_search:
            web_results.extend(search_wikipedia(prompt, lang_code=lang_code, max_results=2))
            web_results.extend(search_duckduckgo_instant(prompt, max_results=2))
            web_results.extend(search_google_news_rss(prompt, lang_code=lang_code, max_results=3))

        # Deduplicate results by title/snippet
        seen_titles: set[str] = set()
        unique_web: list[dict[str, str]] = []
        for item in web_results:
            key = item["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_web.append(item)

        # 4. Synthesize natural answer with LLM + Web Context + Memories + RAG
        synthesized = query_free_llm_with_context(
            prompt,
            lang_code=lang_code,
            web_results=unique_web,
            history=history,
            api_key=gemini_api_key,
            user_memories=user_memories,
            rag_chunks=rag_chunks,
        )

        if synthesized:
            if unique_web:
                src_header = (
                    "\n\n---\n🌐 **المصادر الحية (Live Web Sources):**\n"
                    if lang_code == "ar"
                    else ("\n\n---\n🌐 **Sources Web en direct :**\n" if lang_code == "fr" else "\n\n---\n🌐 **Live Web Sources:**\n")
                )
                links = "\n".join(
                    f"- [{r['source']}: {r['title']}]({r['url']})" if r.get("url") else f"- {r['source']}: {r['title']}"
                    for r in unique_web[:4]
                )
                final_reply = synthesized + src_header + links
            else:
                final_reply = synthesized

            return {
                "reply": final_reply,
                "language": lang_code,
                "task": "live_web_and_ai_qa",
                "response_source": "live_web_ai",
                "citations": unique_web[:4],
            }

        # 5. Pure Web Search Synthesis fallback (works even if external LLM API is unreachable)
        if unique_web:
            top = unique_web[0]
            if lang_code == "ar":
                lines = [
                    f"🔎 **نتيجة البحث المباشر حول «{_clean_search_query(prompt)}»:**",
                    "",
                    f"{top['snippet']}",
                ]
                if len(unique_web) > 1:
                    lines.append("\n**نتائج وأخبار إضافية من الويب وقوقل:**")
                    for r in unique_web[1:4]:
                        lines.append(f"- **{r['title']}** ({r['source']}): {r['snippet']}")
                lines.append("\n🌐 **الروابط والمصادر:**")
                for r in unique_web[:4]:
                    if r.get("url"):
                        lines.append(f"- {r['title']}: {r['url']}")
            elif lang_code == "fr":
                lines = [
                    f"🔎 **Résultat de recherche en direct pour « {_clean_search_query(prompt)} » :**",
                    "",
                    f"{top['snippet']}",
                ]
                if len(unique_web) > 1:
                    lines.append("\n**Autres résultats Web & Google Actualités :**")
                    for r in unique_web[1:4]:
                        lines.append(f"- **{r['title']}** ({r['source']}) : {r['snippet']}")
            else:
                lines = [
                    f"🔎 **Live Web Search Result for '{_clean_search_query(prompt)}':**",
                    "",
                    f"{top['snippet']}",
                ]
                if len(unique_web) > 1:
                    lines.append("\n**Additional Web & Google News Results:**")
                    for r in unique_web[1:4]:
                        lines.append(f"- **{r['title']}** ({r['source']}): {r['snippet']}")

            return {
                "reply": "\n".join(lines),
                "language": lang_code,
                "task": "live_web_search",
                "response_source": "live_web_search",
                "citations": unique_web[:4],
            }

        # 6. Final offline fallback
        return {
            "reply": local_res["reply"],
            "language": lang_code,
            "task": local_res["task"],
            "response_source": "rule_based_fallback_assistant",
            "citations": [],
        }
