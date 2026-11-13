# Legal Source Review

Reviewed on 2026-11-13. Registry entries rechecked: 12. No source was downloaded or imported during this review.

This is an engineering-side review of official source/license statements, not legal advice. The license for a package or repository is not treated as a license for all underlying content.

| Source | Review state | License / evidence result | Internal / local training | Commercial / redistribution |
|---|---|---|---|---|
| tatoeba_cc0 | VERIFIED | CC0-1.0, limited to sentences in the dedicated sentences_CC0 export | Permitted under CC0 for the dedicated export. | Permitted under CC0 for the dedicated export. / Permitted under CC0 for the dedicated export. |
| tatoeba_general | RESTRICTED | Tatoeba states general files are CC BY 2.0 FR; some sentences are separately CC0. | Potentially possible subject to license scope, attribution obligations, and local law; not approved here. | CC BY permits commercial use when license terms are met; per-record scope must be respected. / Requires attribution and intact notices; license and record scope must be preserved. |
| wikimedia | RESTRICTED | Typically CC BY-SA 4.0 and GFDL for contributor text, with project/revision and third-party exceptions. | Not approved until revision-level scope and obligations are implemented. | Licenses generally permit with compliance, attribution, and share-alike; individual exceptions remain. / Attribution and share-alike/GFDL terms can apply; exclude fair-use/third-party material. |
| oscar | UNVERIFIED | Official OSCAR card says CC0 applies only to metadata and annotations; crawled text is third-party content. | Not approved; underlying rights are unresolved and Common Crawl terms include AI/ML training-related indemnity language. | Unknown for crawled contents. / Not established; content-owner terms may separately apply. |
| common_voice_ar_27 | RESTRICTED | Dataset page identifies CC0-1.0 and separately forbids speaker identification and dataset re-hosting/re-sharing. | Local ASR training is within stated use; general LLM text pretraining is not established by this dataset-specific purpose. | CC0 is listed, but platform agreement and usage restrictions must also be followed; not cleared for this project. / Re-hosting/re-sharing prohibited by the dataset page/platform FAQ. |
| common_voice_en_au_24 | RESTRICTED | Dataset page identifies CC0-1.0; dataset page forbids speaker identification and re-hosting/re-sharing. | Local ASR use is stated; general text-only LLM pretraining is not established. | CC0 listed; platform terms and access conditions still apply, so no project approval. / Re-hosting/re-sharing prohibited by the dataset page/platform FAQ. |
| common_voice_fr_26 | RESTRICTED | Dataset page identifies CC0-1.0 and prohibits speaker identification; MDC Common Voice FAQ says datasets may not be re-hosted/shared. | Local speech technology training intended; general LLM text pretraining not established. | CC0 listed, subject to platform agreement and restriction review; not approved. / Re-hosting/re-sharing prohibited by the platform FAQ. |
| madar_parallel | RESTRICTED | Official grant is internal research/evaluation only; no sublicense or further distribution and no modification; commercial rights require separate agreement. | Not established; no modification right is granted; obtain written permission before using for model training. | Not permitted under the displayed grant; separate commercial license is offered. / Not permitted under the displayed grant. |
| culturax | UNVERIFIED | Dataset card says terms strictly follow mC4 and OSCAR; no standalone license grants rights to all underlying web content. | Not approved; underlying crawl-page rights remain unresolved and Common Crawl terms expressly address AI/ML use. | Unknown for underlying webpage content. / Not established for underlying content. |
| oscar_moscar | UNVERIFIED | mOSCAR page licenses actual packaging under CC BY 4.0 and states the project does not own extracted text. | Not approved; CC BY applies to packaging only; source-page rights unresolved. | Unknown for underlying text; packaging license alone is insufficient. / Packaging may be redistributed with CC BY attribution; underlying text requires source-level rights. |
| mot_voa | RESTRICTED | MOT repository states the collection is CC BY 4.0 and source material is public domain. | Potentially permitted with attribution; no project approval until a specific release and file provenance are reviewed. | CC BY 4.0 allows commercial reuse with conditions; confirm each collected artifact is in-scope. / Allowed under CC BY attribution and notice obligations, subject to source material scope. |
| hplt_v1_2 | UNVERIFIED | Official HPLT terms say package is CC0, but HPLT does not own the extracted text; user is responsible for applicable legal framework. v1.2 is marked deprecated. | Not approved; package CC0 does not resolve underlying source rights. | Unknown for underlying texts. / Package license does not resolve redistribution rights for source texts. |

## Source-specific official evidence

### tatoeba_cc0 — VERIFIED
- Publisher: Tatoeba Association.
- Source: https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2
- Evidence: https://tatoeba.org/en/downloads, https://creativecommons.org/publicdomain/zero/1.0/legalcode.en.
- Access: Public weekly export; distinct from general sentences export.
- Research/internal: Permitted for the dedicated CC0 records.
- Local training: Permitted under CC0 for the dedicated export.
- Commercial: Permitted under CC0 for the dedicated export.
- Redistribution/modification: Permitted under CC0 for the dedicated export. / Permitted under CC0.
- Attribution: Not required by CC0; preserving source provenance remains recommended.
- Privacy/geography: Sentence-level corpus; source terms do not establish that every text is free of privacy concerns. / No geographic restriction stated by CC0; applicable law remains relevant.
- NourAI approval: Existing registry says approved; this is the only imported source.
- Unresolved: No independent legal counsel opinion.

### tatoeba_general — RESTRICTED
- Publisher: Tatoeba Association.
- Source: https://downloads.tatoeba.org/exports/sentences.tar.bz2
- Evidence: https://tatoeba.org/en/downloads, https://tatoeba.org/en/terms_of_use, https://creativecommons.org/licenses/by/2.0/fr/legalcode.fr.
- Access: Public download.
- Research/internal: Reuse can be allowed subject to applicable per-record license and attribution terms; no corpus approved.
- Local training: Potentially possible subject to license scope, attribution obligations, and local law; not approved here.
- Commercial: CC BY permits commercial use when license terms are met; per-record scope must be respected.
- Redistribution/modification: Requires attribution and intact notices; license and record scope must be preserved. / Permitted subject to attribution and license terms.
- Attribution: Required for CC BY records; maintain author/title/URI and modification notices where applicable.
- Privacy/geography: Not individually reviewed. / French association terms refer to French, residence, and other applicable laws.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### wikimedia — RESTRICTED
- Publisher: Wikimedia Foundation and individual project contributors.
- Source: https://dumps.wikimedia.org/
- Evidence: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use, https://meta.wikimedia.org/wiki/How_to_use_or_reuse_our_content.
- Access: Public dumps; pin a project and dump date.
- Research/internal: Possible only after per-revision license/exception filtering and compliance review.
- Local training: Not approved until revision-level scope and obligations are implemented.
- Commercial: Licenses generally permit with compliance, attribution, and share-alike; individual exceptions remain.
- Redistribution/modification: Attribution and share-alike/GFDL terms can apply; exclude fair-use/third-party material. / Permitted under compatible license terms where source content allows.
- Attribution: Required; source page authorship/links and modifications must be attributed.
- Privacy/geography: Contributor and third-party content may include personal data; no project-specific review performed. / Applicable laws vary; not individually reviewed.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### oscar — UNVERIFIED
- Publisher: OSCAR / Inria ALMAnaCH; underlying content originates in Common Crawl.
- Source: https://huggingface.co/datasets/oscar-corpus/oscar
- Evidence: https://huggingface.co/datasets/oscar-corpus/oscar, https://commoncrawl.org/terms-of-use.
- Access: Gated access; official card states access is suspended pending clarification.
- Research/internal: Not established for this project; comply with local TDM/research exceptions and underlying terms.
- Local training: Not approved; underlying rights are unresolved and Common Crawl terms include AI/ML training-related indemnity language.
- Commercial: Unknown for crawled contents.
- Redistribution/modification: Not established; content-owner terms may separately apply. / Not established for underlying texts.
- Attribution: Depends on each underlying source; not resolved by packaging.
- Privacy/geography: Official terms warn crawled content may contain personal/sensitive material; privacy risks require review. / Local law materially affects permissible TDM/research use.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### common_voice_ar_27 — RESTRICTED
- Publisher: Mozilla Common Voice; distributed through Mozilla Data Collective.
- Source: https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa
- Evidence: https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa, https://mozilladatacollective.com/terms/consumers, https://community.mozilladatacollective.com/faq-why-cant-i-re-host-or-share-common-voice-datasets-that-i-download-from-mdc/.
- Access: MDC account/platform terms apply; dataset is an ASR audio/transcript release.
- Research/internal: ASR/local research use is the stated purpose, subject to account terms and no speaker identification.
- Local training: Local ASR training is within stated use; general LLM text pretraining is not established by this dataset-specific purpose.
- Commercial: CC0 is listed, but platform agreement and usage restrictions must also be followed; not cleared for this project.
- Redistribution/modification: Re-hosting/re-sharing prohibited by the dataset page/platform FAQ. / CC0 applies to covered dataset content, subject to platform/dataset restrictions.
- Attribution: CC0 does not require attribution; provenance recommended.
- Privacy/geography: No speaker re-identification; speaker metadata and recordings require careful handling. / No dataset-specific geographic license limit found; platform terms and local law apply.
- NourAI approval: Registry says not approved; no data downloaded or ingested.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### common_voice_en_au_24 — RESTRICTED
- Publisher: Mozilla Common Voice; Mozilla Data Collective dataset curator.
- Source: https://mozilladatacollective.com/datasets/cmko7havo02f5nw07rbwwhowe
- Evidence: https://mozilladatacollective.com/datasets/cmko7havo02f5nw07rbwwhowe, https://mozilladatacollective.com/terms/consumers, https://community.mozilladatacollective.com/faq-why-cant-i-re-host-or-share-common-voice-datasets-that-i-download-from-mdc/.
- Access: MDC account/platform terms apply; CSV and MP3 subset.
- Research/internal: ASR/local research use; page says designed for ASR acoustic prediction on Australian English.
- Local training: Local ASR use is stated; general text-only LLM pretraining is not established.
- Commercial: CC0 listed; platform terms and access conditions still apply, so no project approval.
- Redistribution/modification: Re-hosting/re-sharing prohibited by the dataset page/platform FAQ. / CC0 for covered contents, subject to platform restrictions.
- Attribution: CC0 does not require attribution; provenance recommended.
- Privacy/geography: No speaker identification; metadata includes self-described accent and may include other speaker metadata. / No dataset-specific geographic limit found; platform terms and law apply.
- NourAI approval: Registry says not approved; no data downloaded or ingested.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### common_voice_fr_26 — RESTRICTED
- Publisher: Mozilla Common Voice; Mozilla Data Collective curator.
- Source: https://mozilladatacollective.com/datasets/cmrumhinj0001md07y3huuhs5
- Evidence: https://mozilladatacollective.com/datasets/cmrumhinj0001md07y3huuhs5, https://mozilladatacollective.com/terms/consumers, https://community.mozilladatacollective.com/faq-why-cant-i-re-host-or-share-common-voice-datasets-that-i-download-from-mdc/.
- Access: MDC account/platform terms apply; MP3 and TSV.
- Research/internal: Speech technology/linguistics research stated; use subject to platform terms.
- Local training: Local speech technology training intended; general LLM text pretraining not established.
- Commercial: CC0 listed, subject to platform agreement and restriction review; not approved.
- Redistribution/modification: Re-hosting/re-sharing prohibited by the platform FAQ. / CC0 for covered content subject to restrictions.
- Attribution: CC0 does not require attribution; provenance recommended.
- Privacy/geography: No speaker identification. / No dataset-specific geographic limit found; platform terms and law apply.
- NourAI approval: Registry says not approved; no data downloaded or ingested.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### madar_parallel — RESTRICTED
- Publisher: CMU / NYU Abu Dhabi MADAR project.
- Source: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/
- Evidence: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/.
- Access: Download is conditioned on accepting the license; English and French versions require U-Star Consortium contact.
- Research/internal: Permitted only for internal research and evaluation as stated.
- Local training: Not established; no modification right is granted; obtain written permission before using for model training.
- Commercial: Not permitted under the displayed grant; separate commercial license is offered.
- Redistribution/modification: Not permitted under the displayed grant. / Not permitted under the displayed grant.
- Attribution: Citation is requested by project page; license notice must be retained in internal copies.
- Privacy/geography: No privacy statement assessed in the displayed license. / No explicit geographic restriction stated.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### culturax — UNVERIFIED
- Publisher: CulturaX dataset authors / UONLP project; content derives from mC4 and OSCAR.
- Source: https://huggingface.co/datasets/uonlp/CulturaX
- Evidence: https://huggingface.co/datasets/uonlp/CulturaX, https://commoncrawl.org/terms-of-use.
- Access: Hugging Face access may require authentication.
- Research/internal: Not established without underlying source and local-law review.
- Local training: Not approved; underlying crawl-page rights remain unresolved and Common Crawl terms expressly address AI/ML use.
- Commercial: Unknown for underlying webpage content.
- Redistribution/modification: Not established for underlying content. / Not established for underlying content.
- Attribution: Source-specific, not resolved by CulturaX packaging.
- Privacy/geography: Official card warns personal/sensitive information may remain. / Underlying rights and text/data-mining law vary by location.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### oscar_moscar — UNVERIFIED
- Publisher: Inria OSCAR project; underlying text from web pages and Common Crawl.
- Source: https://oscar-project.github.io/documentation/versions/mOSCAR/
- Evidence: https://oscar-project.github.io/documentation/versions/mOSCAR/, https://commoncrawl.org/terms-of-use.
- Access: Access via Hugging Face dataset repository.
- Research/internal: Not established for underlying text absent page-level rights review.
- Local training: Not approved; CC BY applies to packaging only; source-page rights unresolved.
- Commercial: Unknown for underlying text; packaging license alone is insufficient.
- Redistribution/modification: Packaging may be redistributed with CC BY attribution; underlying text requires source-level rights. / Packaging may be modified subject to CC BY; underlying text depends on original source.
- Attribution: CC BY attribution applies to packaging; content may have separate conditions.
- Privacy/geography: Web documents/URLs can reveal personal or sensitive content; no per-record screening performed. / Underlying source rights vary.
- NourAI approval: Not approved for ingestion.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### mot_voa — RESTRICTED
- Publisher: Brandeis University Broadening Linguistic Technologies Lab; source material collected from Voice of America.
- Source: https://github.com/bltlab/mot
- Evidence: https://github.com/bltlab/mot.
- Access: Public GitHub releases; pin the exact release/site files before use.
- Research/internal: Permitted subject to CC BY attribution and exact release validation.
- Local training: Potentially permitted with attribution; no project approval until a specific release and file provenance are reviewed.
- Commercial: CC BY 4.0 allows commercial reuse with conditions; confirm each collected artifact is in-scope.
- Redistribution/modification: Allowed under CC BY attribution and notice obligations, subject to source material scope. / Allowed under CC BY with indication of changes and attribution.
- Attribution: Required; cite MOT and authors and preserve license/changes notice.
- Privacy/geography: Review individual VOA source files and any embedded metadata before use. / No specific geographic restriction stated by repository.
- NourAI approval: Not approved for ingestion; no files downloaded.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

### hplt_v1_2 — UNVERIFIED
- Publisher: HPLT consortium; source collections include Common Crawl and Internet Archive.
- Source: https://hplt-project.org/datasets/v1.2
- Evidence: https://hplt-project.org/datasets/v1.2, https://commoncrawl.org/terms-of-use.
- Access: Public shards; v1.2 deprecated. Current Arabic raw variant is listed at about 174 GB.
- Research/internal: Not established for underlying texts absent source-level rights review.
- Local training: Not approved; package CC0 does not resolve underlying source rights.
- Commercial: Unknown for underlying texts.
- Redistribution/modification: Package license does not resolve redistribution rights for source texts. / Package may be CC0, but underlying text rights unresolved.
- Attribution: Depends on underlying sources; HPLT asks for research citation.
- Privacy/geography: Privacy/anonymization and takedown risks require source-level review. / Applicable text/data mining and privacy laws vary; page explicitly assigns compliance responsibility to user.
- NourAI approval: Not approved; no files downloaded.
- Unresolved: No independent legal counsel opinion.; No individual record-level license audit unless specified.

## Decision

Only `tatoeba_cc0` is both imported and currently approved in the registry. Candidate corpus sources remain unapproved. Review state counts: {'VERIFIED': 1, 'RESTRICTED': 7, 'UNVERIFIED': 4, 'REJECTED': 0}.
