# Similarity Review Decisions

All 15 delegated pairs were reviewed. Twelve cross-split near duplicates were removed only from derived Validation/Test files (5 Validation, 7 Test). Three normalized pairs with an unmapped counterpart were kept; Training and original corpus rows were not changed.

| Pair | Classification | Splits | Decision | Removed ID | Reason |
|---|---|---|---|---|---|
| 10516895 / 10516920 | NEAR_DUPLICATE | train / validation | drop_from_validation_test | 10516920 | The validation sentence adds only hier to the same French proposition; retain the train example and remove the near-copy from validation. |
| 10678217 / 12026764 | NEAR_DUPLICATE | train / test | drop_from_validation_test | 12026764 | The test sentence repeats the same claim with a shortened name; retain train, remove the near-copy from test. |
| 11267026 / 11267027 | NEAR_DUPLICATE | validation / train | drop_from_validation_test | 11267026 | Word order changes only; same English imperative. Remove validation duplicate, retain train. |
| 11502922 / 9937340 | NEAR_DUPLICATE | test / train | drop_from_validation_test | 11502922 | Punctuation-only difference; same sentence. Remove test duplicate, retain train. |
| 12704941 / 12736497 | NEAR_DUPLICATE | test / train | drop_from_validation_test | 12704941 | Same French proposition with one intensifying adjective; avoid near-copy in test, retain train. |
| 12749317 / 12749455 | NEAR_DUPLICATE | train / validation | drop_from_validation_test | 12749455 | Same French question with only personnelle added; remove validation near-copy. |
| 12749347 / 12749423 | NEAR_DUPLICATE | train / test | drop_from_validation_test | 12749423 | Same French question with a short adjective extension; remove test near-copy. |
| 8053062 / 8053063 | NEAR_DUPLICATE | validation / train | drop_from_validation_test | 8053062 | Same invitation with an added vocative; remove validation near-copy to avoid template overlap. |
| 8308080 / 8308081 | NEAR_DUPLICATE | test / train | drop_from_validation_test | 8308080 | Test adds a short preface to the same French sentence; remove the near-copy from test. |
| 8511506 / 8511637 | NORMALIZED_DUPLICATE | train / test | drop_from_validation_test | 8511637 | Same French dialogue with typography and spacing variants; remove test copy, retain train. |
| 8719724 / 8719725 | NEAR_DUPLICATE | validation / train | drop_from_validation_test | 8719724 | Same fable sentence with only the final object omitted; remove validation near-copy. |
| 9182687 / 9182700 | NORMALIZED_DUPLICATE | test / train | drop_from_validation_test | 9182687 | Same greeting with ellipsis/period punctuation variation; remove test copy, retain train. |
| 9860660 / 11087111 | NORMALIZED_DUPLICATE | train / UNMAPPED | keep | — | The second source record is not present in Train/Validation/Test, so no split leakage exists; preserve imported source records and make no data change. |
| 10157781 / 11558339 | NORMALIZED_DUPLICATE | train / UNMAPPED | keep | — | The second source record is not present in Train/Validation/Test, so no split leakage exists; preserve imported source records and make no data change. |
| 11682587 / 11682588 | NORMALIZED_DUPLICATE | train / UNMAPPED | keep | — | The second source record is not present in Train/Validation/Test, so no split leakage exists; preserve imported source records and make no data change. |

## Post-review full detector scan

The existing detector indexed 55,704 records and produced 35 candidates at threshold 0.85. Mapping IDs to actual reviewed split files showed all 35 are Train–Train; there are zero cross-split candidates. All were retained because they are outside the delegated 15 and do not constitute split leakage. LSH may miss pairs; scores are surface similarity rather than semantic proof.
