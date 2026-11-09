# Diagnostics from saved outputs

This folder contains post hoc P01 and P04 diagnostics. It also links to my visual review of the saved edits and textured drafts. The diagnostics use the original model outputs and scores.

Run `python -B additional_analysis/rebuild.py` from the evaluation root. This reads saved outputs and writes only this folder's diagnostic CSVs. It needs the original analysis requirements. It does not call a model, use a GPU, repair a response, or alter the historical scorer.

- `P01_validation.csv`: each original response, extraction availability, historical direct-JSON flag and a separate exact-one-key validator.
- `P01_per_product.csv`: five products, three questions per product/model. The label-only constant answers black, 1 and not_visible match 11 of the 15 labels. This shortcut was identified after the experiment, not run as an original benchmark. It has not viewed images.
- `P04_content_review.csv`: all 24 final traces, with separate protocol, factual, decision and tool-use observations. These interpretations overlap and do not replace the original 1/12 strict score per model. Proposed calls inside invalid replies are not executed calls.

The historical P01 flag in `a1/models.py` checks that `json.loads(raw)["answer"]` is a string. It does not enforce exactly one key or reject duplicate keys. The revised diagnostic validator here does both. It gives 8/15 for A and 0/15 for B on these saved outputs, but this agreement does not make the historical flag a complete schema validator.

The historical extractor in `a1/semantic_answers.py` also accepts a plain-text branch for a non-fenced string not starting with `{` or `[`. The policy name and older prose omit that branch. The complete implemented policy is direct exact-answer JSON, one conventional JSON fence, OR plain text passed unchanged to the answer matcher; malformed JSON is rejected. None of the 30 current final responses uses the plain-text branch, so correcting the description changes no saved score. No historic implementation or policy identifier has been rewritten.

Extraction success is 14/15 for A and 15/15 for B. Supported extracted answers match the labels for every extractable response. The remaining A reply visibly expresses count 1 using single-quote pseudo-JSON; this is readable content, but fails extraction. This does not prove image reasoning: the shortcut, convenience sample and repeated questions limit that interpretation.

[Visual review](student_review/EDITED_REVIEW.html) gives my P02 and P03 observations by job. I recorded 93 passes and 15 issues across 108 judgments. The draft scores and the dimension checks measure different things. The [CSV](student_review/student_review.csv) and [JSON](student_review/student_review.json) contain the complete records, original comments and supporting view descriptions.

`PROPOSED_PROTOCOL.md` describes possible follow-up tests. Those tests would require new inputs and separately recorded outputs.
