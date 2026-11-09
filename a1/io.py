# Read and write project files, checksums and append-only attempt logs.
# These shared helpers keep paths and evidence records consistent.
# Checksums detect changed bytes; they do not establish output quality.

"""Portable files, fingerprints, and append-only attempt records."""

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# Use a timestamp with an explicit time zone.
# Logs from different machines can then be put in the right order.
def utc_now():
    """Return the current UTC time as text for experiment records."""
    return datetime.now(timezone.utc).isoformat()


# Read one saved JSON record.
# Using the same reader keeps configuration and evidence handling consistent.
def read_json(path):
    """Read path as UTF-8 JSON and return its Python value."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


# Write settings in one stable JSON format.
# Equivalent settings should produce the same fingerprint.
def canonical(value):
    # Reject NaN rather than writing nonstandard JSON into scientific records.
    """Convert value to JSON text with a fixed key order and spacing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# Make a checksum of a structured settings record.
# The checksum gives each planned job a link to its exact settings.
def fingerprint(value):
    """Return the SHA-256 checksum of a structured settings value."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# Read a file in chunks and calculate its checksum.
# This checks large artifacts without loading the entire file into memory.
def sha256(path):
    """Return a file's SHA-256 checksum without changing the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


# Resolve a saved relative path inside the project folder.
# Imported records must not point to unrelated files outside the project.
def project_path(root, relative):
    """Return relative resolved inside the project folder root."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves the project: {relative}")
    return path


# Write a complete JSON file before replacing the previous copy.
# An interruption should not leave a half written configuration or evidence file.
def atomic_json(path, value):
    """Save value as readable JSON at path, replacing a complete old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            # Flush the attempt record before starting the next job.
            os.fsync(stream.fileno())
        # Replace the old file only when the new JSON is complete.
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


# Read every line of an attempt log and report malformed records.
# Silently skipping a failed or damaged record would change the evaluation history.
def read_attempts(path):
    """Read path as one JSON attempt record per line and return a list."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed attempt log {path}, line {number}"
                ) from error
    return rows


# Add one record to the end of the attempt log and flush it to disk.
# Earlier attempts remain visible if a later attempt fails or is interrupted.
def append_attempt(path, row):
    """Append row to the attempt log at path and return None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(row) + "\n")
        stream.flush()
        # Flush the attempt record before starting the next job.
        os.fsync(stream.fileno())


# Write a table with a fixed column order.
# The report worksheets remain readable and consistent across runs.
def write_csv(path, rows, fields):
    """Write rows to path using fields as the CSV column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# Save each required file path, size and checksum.
# Each attempt can then be linked to the actual input bytes used.
def file_records(root, paths):
    """Return paths, checksums and byte sizes for the requested input files."""
    records = []
    for relative in sorted(set(paths)):
        path = project_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {relative}")
        records.append(
            {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
        )
    return records
