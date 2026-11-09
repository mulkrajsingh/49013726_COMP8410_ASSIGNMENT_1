# Read saved file records and the export index; verify their paths and bytes.
# Use preserved execution-source copies for historical code when explicitly mapped.
# Current images and catalogue inputs must still match their recorded checksums.

"""Check saved runs against the files that were used during execution."""

from pathlib import Path

from .io import project_path, read_json, sha256


# Read the small export index once per project. It explains metadata edits and
# points to the original execution source, rather than the edited teaching copy.
def evidence_index(root):
    path = Path(root) / "evidence/submission_export.json"
    return read_json(path) if path.exists() else {"files": [], "execution_sources": []}


# Scientific inputs and generated outputs must still match their original bytes.
# Only listed metadata exports may use a different, explicitly recorded checksum.
def verified_file(root, record, source=False):
    index = evidence_index(root)
    relative = record["path"]
    if source and relative.endswith(".py"):
        for item in index["execution_sources"]:
            if item["path"] == relative and item["sha256"] == record["sha256"]:
                path = project_path(root, item["saved_path"])
                if path.is_file() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]:
                    return path
                raise ValueError("Execution source differs: " + relative)
    path = project_path(root, relative)
    if not path.is_file():
        raise ValueError("Missing saved file: " + relative)
    actual = sha256(path)
    if actual == record["sha256"] and path.stat().st_size == record["bytes"]:
        return path
    for item in index["files"]:
        if item["path"] == relative and item["original_sha256"] == record["sha256"]:
            if item["kind"] != "metadata_export" or path.suffix not in {".json", ".jsonl", ".txt", ".log", ".csv", ".md", ".html"}:
                raise ValueError("Unsupported evidence export: " + relative)
            if actual == item["submission_sha256"] and path.stat().st_size == item["submission_bytes"]:
                return path
    raise ValueError("Saved file checksum differs: " + relative)


# Keep the current input files in the comparison. A changed image or catalogue
# must invalidate the saved run even when an execution-source copy is available.
def verify_inputs(root, job, record):
    expected_paths = sorted(job["item"]["files"])
    if sorted(item["path"] for item in record["input_files"]) != expected_paths:
        raise ValueError("Recorded input paths differ")
    for item in record["input_files"]:
        verified_file(root, item, source=True)

