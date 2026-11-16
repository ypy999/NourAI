from nourai.engines import NourAIPlatform


def test_nourai_8_layer_engines():
    platform = NourAIPlatform()

    # 1. Foundation Model status check
    status = platform.foundation_model.status()
    assert status["name"] == "NourAI Foundation Model"
    assert status["ready"] is True

    # 2. Reasoning Engine: separates normal coding/planning question from web search
    code_trace = platform.reasoning.analyze("اكتب دالة بايثون لفرز قائمة")
    assert code_trace.intent == "coding"
    assert code_trace.needs_search is False

    search_trace = platform.reasoning.analyze("ابحث عن تاريخ الذكاء الاصطناعي")
    assert search_trace.intent == "search"
    assert search_trace.needs_search is True

    # 3. Memory Engine: extracts and recalls user facts
    added = platform.memory.extract_and_store("اسمي أحمد وأعمل كمطور برمجيات")
    assert len(added) >= 1
    assert any("أحمد" in m for m in platform.memory.recall())

    # 4. Knowledge Engine: indexes & retrieves relevant chunks
    platform.knowledge.add_document("guide.txt", "NourAI يدعم العربية والإنجليزية والفرنسية مع وضع المطور.")
    hits = platform.knowledge.retrieve("هل يدعم العربية")
    assert len(hits) == 1
    assert hits[0]["filename"] == "guide.txt"

    # 5. Coding Engine: syntax validation
    assert platform.coding.check_python_syntax("def f(x):\n    return x + 1")["valid"] is True
    assert platform.coding.check_python_syntax("def f(x:\n    return")["valid"] is False

    # 6. Agent Engine + 7. Evaluation Engine via unified platform
    res = platform.process_turn("احسب 25 * 4 + 10")
    assert res["tool_math"] == "110"
    assert res["evaluation"]["passed"] is True
    assert "110" in res["evaluation"]["cleaned_answer"]
