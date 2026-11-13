# Document Leakage Policy

Status: NOT_CHECKABLE for current Tatoeba corpus. The raw archive contains the CC0 sentences export only; the inspected record schema has sentence IDs and language/text/timestamp but no document IDs or translation-group/link table. Sentence IDs identify individual sentences, not document membership. No document or translation-group IDs are inferred.

Current closure checks exact source_record_id, exact text and normalized text leakage. Near-duplicate screening is a candidate review, not proof of document isolation. Any future source whose records represent documents or translation groups must provide a genuine document_id/group_id or be rejected/held UNKNOWN before splitting. Sentence-level sources may use sentence-level split policy only when that limitation is explicitly accepted; it does not pass document leakage.
