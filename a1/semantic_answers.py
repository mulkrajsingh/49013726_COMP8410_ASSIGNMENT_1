# Read a saved P01 reply; return an extracted answer and its extraction method.
# Accept only the implemented wrappers or the unchanged plain-text branch.
# Keep the original response and historical JSON-format flag unchanged.

"""Read saved P01 text for semantic analysis without changing generation files.

Keep generation files and their strict JSON flag unchanged. Read one saved
answer and remove only an exact answer JSON object or one JSON Markdown fence.
Plain answers still go to the existing accepted-answer matcher unchanged.
"""

import hashlib
import json
import re

SEMANTIC_ANALYSIS_POLICY = "exact_answer_json_or_single_json_fence_v1"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("non-standard JSON constant: " + value)


def semantic_answer(saved):
    """Return a semantic value and its provenance without changing saved data."""
    source_field = "raw_text" if isinstance(saved.get("raw_text"), str) else "answer"
    source = saved.get(source_field)
    result = {
        "semantic_answer": None,
        "semantic_value_available": False,
        "semantic_extraction_method": "rejected",
        "semantic_source_field": source_field,
        "semantic_source_sha256": None,
        "semantic_rejection_reason": None,
        "structured_parse_success": saved.get("structured_parse_success"),
        "structured_format_failure": saved.get("structured_parse_success") is False,
    }
    if not isinstance(source, str):
        result["semantic_rejection_reason"] = "source is not a string"
        return result
    result["semantic_source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    text = source.strip()
    method = "direct_json"
    if text.startswith("```"):
        fence = re.fullmatch(r"```json[ \t]*\r?\n(.*?)\r?\n```", text, flags=re.S)
        if fence is None:
            result["semantic_rejection_reason"] = "not one conventional json fence"
            return result
        text = fence.group(1)
        method = "json_fence"
    elif not text.startswith(("{", "[")):
        # Keep the previous short plain-answer comparison, including numbers.
        result.update(semantic_answer=source, semantic_value_available=True,
                      semantic_extraction_method="plain_text")
        return result
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    except (ValueError, TypeError):
        result["semantic_rejection_reason"] = "invalid or non-unique JSON"
        return result
    if not isinstance(value, dict) or set(value) != {"answer"} or not isinstance(value["answer"], str):
        result["semantic_rejection_reason"] = "requires only an answer field containing a string"
        return result
    result.update(semantic_answer=value["answer"], semantic_value_available=True,
                  semantic_extraction_method=method)
    return result
