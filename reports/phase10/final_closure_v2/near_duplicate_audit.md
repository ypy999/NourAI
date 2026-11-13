# Near-Duplicate Audit

Status: `FULL_CORPUS_CANDIDATE_SCAN_COMPLETED_WITH_LSH_RECALL_LIMITATION`

Records examined: 55,719; invalid: 0; unique source IDs: 55,719.
Exact duplicate groups: 0; affected rows: 0.
Normalized duplicate groups: 3; affected rows: 6; within/between-split normalized duplicate pairs: 0.
Near-duplicate candidates: 50; cross-split candidate pairs requiring review: 12.

## Method

- Exact SHA-256 over UTF-8 text.
- Project normalization profile: NFC, Unicode case-folding, collapsed whitespace; punctuation/diacritics preserved.
- Full corpus inserted into the existing SQLite-backed detector; word-trigram MinHash LSH with 16 permutations, 4 bands, 200 candidate cap and Jaccard threshold 0.85.
- Candidate records are not automatically deleted or filtered.

## Candidate pairs across splits

- `train` line 12084 / corpus line 13347 ↔ `validation` line 647 / corpus line 13348; IDs `10516895` / `10516920`; similarity 0.857143. Texts: 'Voici l’enfant qui a gagné un prix.' ↔ 'Voici l’enfant qui a gagné un prix hier.'
- `train` line 13874 / corpus line 15354 ↔ `test` line 1602 / corpus line 32040; IDs `10678217` / `12026764`; similarity 0.857143. Texts: 'Salvador Dali was kicked out of the surrealist movement.' ↔ 'Dali was kicked out of the surrealist movement.'
- `validation` line 1110 / corpus line 22299 ↔ `train` line 20114 / corpus line 22300; IDs `11267026` / `11267027`; similarity 1.000000. Texts: 'Everybody run!' ↔ 'Run, everybody!'
- `test` line 1207 / corpus line 24533 ↔ `train` line 7500 / corpus line 8286; IDs `11502922` / `9937340`; similarity 1.000000. Texts: 'Beg me to eat!' ↔ 'Beg me to eat.'
- `test` line 2091 / corpus line 41692 ↔ `train` line 37687 / corpus line 41897; IDs `12704941` / `12736497`; similarity 0.875000. Texts: 'La communication authentique aligne nos paroles sur nos valeurs.' ↔ 'La communication authentique aligne nos paroles sur nos valeurs profondes.'
- `train` line 40489 / corpus line 44976 ↔ `validation` line 2250 / corpus line 45078; IDs `12749317` / `12749455`; similarity 0.909091. Texts: "Comment la mémoire épisodique contribue-t-elle à notre sentiment d'identité ?" ↔ "Comment la mémoire épisodique contribue-t-elle à notre sentiment d'identité personnelle ?"
- `train` line 40512 / corpus line 45001 ↔ `test` line 2245 / corpus line 45059; IDs `12749347` / `12749423`; similarity 0.900000. Texts: "Comment l'anthropologie du numérique étudie-t-elle les cultures virtuelles ?" ↔ "Comment l'anthropologie du numérique étudie-t-elle les cultures virtuelles émergentes ?"
- `validation` line 43 / corpus line 1209 ↔ `train` line 1130 / corpus line 1210; IDs `8053062` / `8053063`; similarity 0.888889. Texts: "Why don't you pop in for a drink tonight?" ↔ "Tom, why don't you pop in for a drink tonight?"
- `test` line 77 / corpus line 1912 ↔ `train` line 1758 / corpus line 1913; IDs `8308080` / `8308081`; similarity 0.875000. Texts: 'Tu sais ? Les toponymes que je viens de donner sont ceux de lieux-dits du village Djemaa-Saharidj.' ↔ 'Les toponymes que je viens de donner sont ceux de lieux-dits du village Djemaa-Saharidj.'
- `train` line 2058 / corpus line 2249 ↔ `test` line 92 / corpus line 2252; IDs `8511506` / `8511637`; similarity 1.000000. Texts: '"Qu\'arrivera-t-il à notre langue kabyle?"\xa0 "Elle durera pour toujours!"' ↔ "« Qu'arrivera-t-il à notre langue kabyle ? » « Elle durera pour toujours ! »"
- `validation` line 129 / corpus line 2824 ↔ `train` line 2571 / corpus line 2825; IDs `8719724` / `8719725`; similarity 0.937500. Texts: 'This is the fable of the goat saying to the jackal: "Even when I graze, I watch you!"' ↔ 'This is the fable of the goat saying to the jackal: "Even when I graze, I watch!"'
- `test` line 221 / corpus line 5039 ↔ `train` line 4602 / corpus line 5040; IDs `9182687` / `9182700`; similarity 1.000000. Texts: 'Hi to those over there ...' ↔ 'Hi to those over there.'

LSH is approximate and can miss similar pairs. Candidate pairs are surface-similarity signals, not semantic judgments; human review is required.
