# Read student observations and their cited job IDs; write a feedback summary.
# Check required fields and evidence from both model sides.
# Blank worksheets remain incomplete; this code does not supply student judgments.

"""Genuine student self-evaluation without invented participant evidence."""

import csv
import math
from datetime import datetime
from pathlib import Path

from .io import atomic_json, read_json, write_csv
from .study import planned_jobs
from .worker import successful_record
from .analysis import _last_record

CHOICES = {"A", "B", "tie", "neither", "unable_to_judge"}
FIELDS = [
    "pair",
    "reviewer",
    "evaluated_at",
    "verdict",
    "evidence_job_ids",
    "observations",
    "limitations",
    "review_seconds",
]


# Create the blank student observation worksheet if it does not exist.
# Rerunning preparation must not erase observations already entered by the student.
def prepare_feedback(root):
    """Create blank pair-level observation files under root if missing."""
    root = Path(root)
    path = root / "evidence/self_evaluation/observations.csv"
    if not path.exists():
        rows = []
        for case in read_json(root / "config/study.json")["cases"]:
            rows.append({"pair": case["pair"]})
        write_csv(path, rows, FIELDS)
    guide = path.parent / "README.md"
    if not guide.exists():
        guide.write_text(
            "# Student self evaluation\n\nThe professor permits feedback from your own observations and experiments. Blank rows do not establish completed feedback.\n\nAfter reviewing real outputs, fill one row per active pair with yourself as reviewer, an ISO timestamp, A/B/tie/neither/unable_to_judge, observations and limitations. Use semicolon-separated final job IDs, including both model sides. Include failures and explain example selection. Review every final output in data/manual_reviews.csv.\n\nThis is student self-evaluation with no blinding or independent participant sample. Do not claim recruitment, consent or population preference.\n"
        )
    return {
        "mode": "student_self_evaluation",
        "template": str(path.relative_to(root)),
        "external_participants_required": False,
        "completed_feedback_created": False,
    }


# Check that student feedback cites real final attempts from both models.
# Unrun examples and blank worksheets cannot establish completed feedback.
def summarize_feedback(root):
    """Validate the observation CSV under root and return its summary."""
    root = Path(root)
    path = root / "evidence/self_evaluation/observations.csv"
    result = {
        "mode": "student_self_evaluation",
        "completed_pairs": 0,
        "external_participants": 0,
        "independent_user_study": False,
        "by_pair": {},
        "status": "not_collected",
    }
    if not path.exists():
        return result
    jobs = {}
    for job in planned_jobs(root):
        jobs[job["job_id"]] = job
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    recorded_pairs = set()
    for row in rows:
        recorded_pairs.add(row["pair"])
    expected_pairs = set()
    for case in read_json(root / "config/study.json")["cases"]:
        expected_pairs.add(case["pair"])
    if len(recorded_pairs) != len(rows) or recorded_pairs != expected_pairs:
        raise ValueError("Self-evaluation must have exactly one row per active pair")

    for row in rows:
        pair = row["pair"]
        has_observations = False
        for field in FIELDS:
            if field != "pair" and row.get(field, "").strip():
                has_observations = True
                break
        if not has_observations:
            result["by_pair"][pair] = {"status": "pending"}
            continue
        required_fields = [
            "reviewer",
            "evaluated_at",
            "verdict",
            "evidence_job_ids",
            "observations",
            "limitations",
        ]
        for key in required_fields:
            if not row.get(key, "").strip():
                raise ValueError(f"Incomplete genuine self-evaluation: {pair}")
        if row["verdict"] not in CHOICES:
            raise ValueError("Invalid self-evaluation verdict")
        datetime.fromisoformat(row["evaluated_at"].replace("Z", "+00:00"))
        duration = None
        if row.get("review_seconds"):
            duration = float(row["review_seconds"])
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError("Review duration must be finite and nonnegative")

        # The CSV stores several job IDs separated by semicolons.
        ids = []
        for value in row["evidence_job_ids"].split(";"):
            if value.strip():
                ids.append(value.strip())
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Feedback evidence must use unique final job IDs from this pair"
            )
        sides = set()
        for job_id in ids:
            if job_id not in jobs or jobs[job_id]["pair"] != pair:
                raise ValueError(
                    "Feedback evidence must use unique final job IDs from this pair"
                )
            sides.add(jobs[job_id]["side"])
        if sides != {"A", "B"}:
            raise ValueError("Compare actual evidence for both model sides")
        for job_id in ids:
            job = jobs[job_id]
            last = _last_record(root, job)
            try:
                valid = successful_record(root, job) is not None
            except (OSError, ValueError):
                valid = False
            if not valid and (
                last is None
                or last["status"] not in {"failed", "interrupted", "invalidated"}
            ):
                raise ValueError(
                    "Feedback cannot cite an unrun or corrupted output as reviewed evidence"
                )
        result["by_pair"][pair] = {"status": "recorded", **row}
        result["completed_pairs"] += 1
    if result["completed_pairs"] == len(rows):
        result["status"] = "complete"
    elif result["completed_pairs"]:
        result["status"] = "partial"
    atomic_json(path.parent / "summary.json", result)
    return result
