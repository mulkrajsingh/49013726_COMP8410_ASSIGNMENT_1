# Run CPU fixtures for review handling, answer parsing and evidence validation.
# Temporary inputs test failure boundaries without changing saved experiments.
# A passing software test is not a new model-performance result.

"""Small checks for scoring and saved-evidence handling; no model execution."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.agents import facts_equal, parse_reply
from a1.analysis import _boolean
from a1.evidence import verified_file, verify_inputs
from a1.metrics import checklist_score
from a1.semantic_answers import semantic_answer


class AnalysisChecks(unittest.TestCase):
    # Missing review cells must not silently become successful or failed ratings.
    def test_missing_reviews(self):
        self.assertIsNone(_boolean(""))
        self.assertIsNone(checklist_score([True, None, True]))
        self.assertEqual(checklist_score([True, False, True]), 2 / 3)

    # Read exact JSON or one JSON fence, while preserving the original format flag.
    def test_semantics_and_format_are_separate(self):
        saved = {"raw_text": '```json\n{"answer":"brown"}\n```', "structured_parse_success": False}
        result = semantic_answer(saved)
        self.assertEqual(result["semantic_answer"], "brown")
        self.assertFalse(result["structured_parse_success"])
        self.assertEqual(saved["raw_text"], '```json\n{"answer":"brown"}\n```')

    def test_invalid_json_is_not_repaired(self):
        for text in ["```json\n{'answer':'2'}\n```", '{"answer":"1","answer":"2"}', '{"answer":"2","extra":true}']:
            self.assertFalse(semantic_answer({"raw_text": text})["semantic_value_available"])

    # A correct price with extra keys still fails the exact fact contract.
    def test_strict_agent_contract(self):
        self.assertFalse(facts_equal({"price": 99, "name": "chair"}, {"price": 99}))
        self.assertFalse(facts_equal({"stock": True}, {"stock": 1}))
        self.assertEqual(parse_reply('<tool_call>{"name":"x","arguments":{}}</tool_call> commentary')[0], "invalid")

    def test_original_execution_source_and_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            (root / "input.txt").write_text("original input")
            (root / "source.py").write_text("# clearer comment\nvalue = 1\n")
            original = b"value = 1\n"
            (root / "evidence/source.py").write_bytes(original)
            source = {"path": "source.py", "sha256": hashlib.sha256(original).hexdigest(), "bytes": len(original)}
            index = {"files": [], "execution_sources": [{**source, "saved_path": "evidence/source.py"}]}
            (root / "evidence/submission_export.json").write_text(json.dumps(index))
            data = (root / "input.txt").read_bytes()
            inputs = {"input_files": [source, {"path": "input.txt", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}]}
            job = {"item": {"files": ["source.py", "input.txt"]}}
            verify_inputs(root, job, inputs)
            (root / "input.txt").write_text("changed scientific input")
            with self.assertRaises(ValueError):
                verify_inputs(root, job, inputs)

    def test_metadata_export_does_not_allow_output_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            (root / "output.png").write_bytes(b"changed output")
            record = {"path": "output.png", "sha256": "0" * 64, "bytes": 1}
            index = {"execution_sources": [], "files": [{"path": "output.png", "original_sha256": "0" * 64, "kind": "metadata_export", "submission_sha256": hashlib.sha256(b"changed output").hexdigest(), "submission_bytes": 14}]}
            (root / "evidence/submission_export.json").write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                verified_file(root, record)


if __name__ == "__main__":
    unittest.main()
