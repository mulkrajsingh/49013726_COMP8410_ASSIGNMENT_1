# Read saved attempts, checked labels and recorded reviews; write result tables.
# Keep incomplete feedback and failed attempts visible in the comparison.
# This is CPU analysis of saved evidence, not a new model run.

"""Rebuild transparent CPU tables from actual saved artifacts and human labels."""

import csv
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_json, project_path, read_attempts, read_json, sha256, write_csv
from .media import image_array
from .metrics import (
    checklist_score,
    group_means,
    masked_mae,
    summarize,
    supported_answer,
)
from .semantic_answers import SEMANTIC_ANALYSIS_POLICY, semantic_answer
from .study import planned_jobs
from .worker import log_path, successful_record

MANUAL_EXTRA = [
    "unsupported_assertion",
    "abstention_correct",
    "usefulness",
    "review_seconds",
    "notes",
    "reviewed_at",
    "imports_reviewer",
    "materials_complete_reviewer",
    "colour_fidelity",
    "edit_instruction_reviewer",
    "product_structure_reviewer",
    "unaffected_region_reviewer",
    "visible_attributes_reviewer",
]


def rebuild_resource_table(root):
    """Export every recorded attempt, including pilots, failures and retries."""
    root = Path(root)
    rows = []
    fields = ["job_id", "pair", "side", "phase", "attempt", "status", "started_at",
              "finished_at", "model_reused", "model_load_seconds", "generation_seconds",
              "end_to_end_seconds", "resource_directory", "telemetry_complete", "samples",
              "max_sample_gap_seconds", "process_peak_rss_bytes", "process_peak_cpu_percent",
              "observed_cpu_seconds", "container_peak_memory_bytes", "container_lifetime_peak_bytes",
              "container_memory_limit_bytes", "gpu_largest_device_peak_used_bytes",
              "gpu_peak_utilization_percent", "gpu_peak_power_watts", "gpu_peak_temperature_c",
              "torch_peak_allocated_bytes", "torch_peak_reserved_bytes", "artifact_bytes",
              "telemetry_errors", "error"]
    for log in sorted((root / "outputs").glob("*/*/*/attempts.jsonl")):
        latest = {}
        for record in read_attempts(log):
            latest[(record["job_id"], record["attempt"])] = record
        for record in latest.values():
            row = {key: record.get(key) for key in fields}
            if row["status"] == "started":
                row["status"] = "interrupted"
            resources = record.get("resources", {})
            # A killed worker may leave a sampler summary without a finish log.
            directory = record.get("resource_directory")
            if not resources and directory:
                saved = project_path(root, directory) / "summary.json"
                if saved.exists():
                    resources = read_json(saved)
            process = resources.get("process_peaks", {})
            container = resources.get("container_peaks", {})
            row.update(
                telemetry_complete=resources.get("complete", False),
                samples=resources.get("samples"),
                max_sample_gap_seconds=resources.get("max_sample_gap_seconds"),
                process_peak_rss_bytes=process.get("rss_bytes"),
                process_peak_cpu_percent=process.get("cpu_percent"),
                observed_cpu_seconds=process.get("observed_cpu_seconds"),
                container_peak_memory_bytes=container.get("memory_bytes"),
                container_lifetime_peak_bytes=container.get("lifetime_memory_peak_bytes"),
                container_memory_limit_bytes=container.get("memory_limit_bytes"),
                telemetry_errors="; ".join(resources.get("errors", [])),
            )
            for target, source in [
                ("gpu_largest_device_peak_used_bytes", "memory_used_bytes"),
                ("gpu_peak_utilization_percent", "utilization_percent"),
                ("gpu_peak_power_watts", "power_watts"),
                ("gpu_peak_temperature_c", "temperature_c"),
            ]:
                observed = [gpu[source] for gpu in resources.get("gpu_peaks", {}).values()
                            if gpu.get(source) is not None]
                row[target] = max(observed) if observed else None
            artifacts = record.get("artifacts")
            row["artifact_bytes"] = sum(item["bytes"] for item in artifacts) if artifacts is not None else None
            rows.append(row)
    output = root / "results/resource_usage.csv"
    write_csv(output, rows, fields)
    return output


# Keep missing reviews separate from failed checks.
def _boolean(value):
    """Convert a review cell value to True, False or None."""
    if value is None or value == "":
        return None
    if value is True or str(value).lower() in {"true", "1"}:
        return True
    if value is False or str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Expected true, false, or blank; got {value!r}")


# Create one blank review row per planned final job.
# Existing observations are preserved when the planned job identifiers change.
def review_template(root):
    """Create or reconcile data/manual_reviews.csv and return its path."""
    study = read_json(Path(root) / "config/study.json")
    identity_fields = ["job_id", "pair", "side", "input_id"]
    fields = identity_fields.copy()
    review_checks = set()
    for case in study["cases"]:
        for check in case["scoring"]["checks"]:
            review_checks.add(check)
    fields += sorted(review_checks)
    fields += MANUAL_EXTRA
    rows = []
    for job in planned_jobs(root):
        row = {}
        for key in identity_fields:
            row[key] = job[key]
        rows.append(row)

    path = Path(root) / "data/manual_reviews.csv"
    if path.exists():
        with path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
        old = {}
        for row in existing:
            old[row["job_id"]] = row
        if len(old) != len(existing):
            raise ValueError("Duplicate manual review job ID")
        current = set()
        for row in rows:
            current.add(row["job_id"])
        if set(old) != current:
            # A retired blank row can be replaced; a filled one needs attention.
            for job_id, row in old.items():
                if job_id not in current:
                    for key, value in row.items():
                        if key not in identity_fields and value not in (None, ""):
                            raise ValueError(
                                "Manual review IDs no longer match: preserve and review filled retired rows before rebuilding"
                            )
            archive = (
                Path(root)
                / "evidence/review_template_history"
                / (sha256(path) + ".csv")
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copy2(path, archive)
            if sha256(archive) != sha256(path):
                raise ValueError("Archived review template checksum differs")
            updated_rows = []
            for row in rows:
                updated_rows.append(old.get(row["job_id"], row))
            write_csv(path, updated_rows, fields)
    else:
        write_csv(path, rows, fields)
    return path


# Reject duplicate job IDs so each output has one review record.
def _manual_rows(root):
    """Read the review CSV under root and return rows indexed by job ID."""
    path = Path(root) / "data/manual_reviews.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    reviews = {}
    for row in rows:
        reviews[row["job_id"]] = row
    if len(reviews) != len(rows):
        raise ValueError("Duplicate manual review job ID")
    return reviews


# Treat an unfinished attempt as interrupted, not successful.
def _last_record(root, job):
    """Return the latest attempt for job from the logs under root."""
    attempts = []
    for record in read_attempts(log_path(root, job)):
        if record["job_id"] == job["job_id"]:
            attempts.append(record)
    if not attempts:
        return None
    record = attempts[-1]
    if record["status"] == "started":
        return {**record, "status": "interrupted"}
    return record


# Combine one saved output with its checked labels and human review.
# Each experiment needs its own task score and a clear completion status.
def score_job(root, job, annotations, manual):
    """Return one score row from a saved job, labels and manual review."""
    result = {
        "job_id": job["job_id"],
        "pair": job["pair"],
        "side": job["side"],
        "input_id": job["input_id"],
        "source_group": job["source_group"],
        "seed": job["seed"],
        "status": "pending",
        "primary_score": None,
        "accepted": None,
    }
    last = _last_record(root, job)
    if not last:
        return result
    try:
        record = successful_record(root, job)
    except (FileNotFoundError, ValueError):
        record = None
    if record is None:
        if last["status"] == "success":
            result["status"] = "invalidated"
        else:
            result["status"] = last["status"]
        result["failure_reason"] = last.get(
            "error", "saved artifact or input validation failed"
        )
        return result
    result.update(
        status="success",
        generation_seconds=record.get("generation_seconds"),
        end_to_end_seconds=record.get("end_to_end_seconds"),
        cost=record.get("cost"),
        output_path=record["output_path"],
    )
    output = project_path(root, record["output_path"])
    labels = annotations[job["input_id"]]
    checks = []
    for check in job["scoring"]["checks"]:
        checks.append(_boolean(manual.get(check)))
    # A missing check keeps this score missing until the output is reviewed.
    checklist = checklist_score(checks)
    if job["pair"] == "P01":
        answer = read_json(output)
        # Extract the answer without changing its historical format flag.
        result.update(semantic_answer(answer))
        result["semantic_analysis_policy"] = SEMANTIC_ANALYSIS_POLICY
        if labels.get("reviewed") and labels.get("accepted_answers"):
            result["primary_score"] = (
                supported_answer(result["semantic_answer"], labels)
                if result["semantic_value_available"] else 0.0
            )
            result["accepted"] = (
                bool(result["primary_score"])
                and answer.get("structured_parse_success") is True
            )
        result["structured_parse_success"] = answer.get("structured_parse_success")
        result["possibly_truncated"] = answer.get("possibly_truncated")
    elif job["pair"] == "P04":
        from .agents import score_agent

        if labels.get("reviewed") and labels.get("expected"):
            measures = score_agent(
                read_json(output),
                read_json(project_path(root, job["item"]["catalogue"])),
                labels["expected"],
            )
            result.update(measures)
            result["primary_score"] = float(measures["task_success"])
            result["accepted"] = measures["task_success"]
    else:
        # Acceptance needs every required check to pass.
        # An incomplete review remains missing, rather than becoming a failed score.
        result["primary_score"] = None
        if checklist is not None:
            result["primary_score"] = float(checklist == 1)
        result["accepted"] = None
        if checklist is not None:
            result["accepted"] = bool(checklist == 1)
    if job["pair"] == "P02" and labels.get("reviewed"):
        reference = image_array(project_path(root, job["item"]["image"]))
        prediction = image_array(output)
        with Image.open(project_path(root, job["item"]["mask"])) as mask:
            # This mask scores preservation; it is not supplied to the generator.
            fixed_mask = np.asarray(mask.convert("L")) > 127
        result["unchanged_region_mae"] = masked_mae(reference, prediction, fixed_mask)
    if job["pair"] == "P03":
        validation = record["output_validation"]
        result.update(
            asset_bytes=validation["bytes"],
            asset_faces=validation["faces"],
            asset_textured=validation["textured"],
        )
    for field in ["unsupported_assertion", "abstention_correct", "colour_fidelity"]:
        result[field] = _boolean(manual.get(field))
    # Keep secondary human evidence visible rather than discarding filled columns.
    usefulness = manual.get("usefulness")
    result["usefulness"] = None
    if usefulness not in (None, ""):
        result["usefulness"] = int(usefulness)
    if result["usefulness"] is not None and result["usefulness"] not in range(1, 6):
        raise ValueError("Usefulness must be an integer from 1 to 5, or blank")
    value = manual.get("review_seconds")
    result["review_seconds"] = None
    if value not in (None, ""):
        result["review_seconds"] = float(value)
    if result["review_seconds"] is not None and (
        not math.isfinite(result["review_seconds"]) or result["review_seconds"] < 0
    ):
        raise ValueError("Review duration must be finite and nonnegative")
    result["notes"] = manual.get("notes", "")
    for field in MANUAL_EXTRA:
        if field.endswith("_reviewer") or field == "reviewed_at":
            result[field] = manual.get(field, "")
    for check, value in zip(job["scoring"]["checks"], checks, strict=True):
        result[check] = value
    return result


def _add_p01_semantic_summary(summary, rows):
    """Keep P01 semantic accuracy separate from its original format flags."""
    fields = ["semantic_analysis_policy", "semantic_scored_outputs", "semantic_correct_count",
              "semantic_correct_format_failure_count", "semantic_value_available_count",
              "semantic_direct_json_count", "semantic_json_fence_count", "semantic_plain_text_count",
              "semantic_rejected_count", "strict_format_success_count", "strict_format_failure_count",
              "strict_format_unknown_count", "semantic_correct_all_planned_rate",
              "strict_format_success_all_planned_rate"]
    for model in summary:
        model.update({field: None for field in fields})
        if model["pair"] != "P01":
            continue
        valid = [row for row in rows if row["pair"] == "P01" and row["side"] == model["side"]
                 and row["status"] == "success"]
        scored = [row for row in valid if row.get("primary_score") is not None]
        model["semantic_analysis_policy"] = SEMANTIC_ANALYSIS_POLICY
        model["semantic_scored_outputs"] = len(scored)
        model["semantic_correct_count"] = sum(row["primary_score"] == 1.0 for row in scored)
        model["semantic_correct_format_failure_count"] = sum(
            row["primary_score"] == 1.0 and row.get("structured_parse_success") is False for row in scored)
        model["semantic_value_available_count"] = sum(row.get("semantic_value_available") is True for row in valid)
        for name, method in [("semantic_direct_json_count", "direct_json"),
                             ("semantic_json_fence_count", "json_fence"),
                             ("semantic_plain_text_count", "plain_text"),
                             ("semantic_rejected_count", "rejected")]:
            model[name] = sum(row.get("semantic_extraction_method") == method for row in valid)
        model["strict_format_success_count"] = sum(row.get("structured_parse_success") is True for row in valid)
        model["strict_format_failure_count"] = sum(row.get("structured_parse_success") is False for row in valid)
        model["strict_format_unknown_count"] = sum(type(row.get("structured_parse_success")) is not bool for row in valid)
        if model["pending"] == 0 and model["planned"]:
            if len(scored) == len(valid):
                model["semantic_correct_all_planned_rate"] = model["semantic_correct_count"] / model["planned"]
            if model["strict_format_unknown_count"] == 0:
                model["strict_format_success_all_planned_rate"] = model["strict_format_success_count"] / model["planned"]
    return summary


# Recalculate all result tables from the original saved evidence.
# The report should use reproducible tables instead of manually edited scores.
def rebuild_results(root):
    """Recalculate and save the result tables in the project under root."""
    root = Path(root)
    jobs = planned_jobs(root)
    cases = {}
    for case in read_json(Path(root) / "config/study.json")["cases"]:
        cases[case["pair"]] = case
    labels = {}
    for pair, case in cases.items():
        labels[pair] = read_json(root / case["annotations_file"])
    # Read student judgments without filling in missing observations.
    manual = _manual_rows(root)
    planned_ids = set()
    for job in jobs:
        planned_ids.add(job["job_id"])
    if set(manual) - planned_ids:
        raise ValueError(
            "Manual review rows refer to a different configuration or protocol"
        )
    rows = []
    for job in jobs:
        rows.append(
            score_job(root, job, labels[job["pair"]], manual.get(job["job_id"], {}))
        )
    summary = summarize(jobs, rows)
    _add_p01_semantic_summary(summary, rows)
    folder = root / "results"
    status = "pending"
    for row in rows:
        if row["status"] == "success":
            status = "partial_or_complete_actual_evidence"
            break
    atomic_json(
        folder / "results.json",
        {
            "jobs": rows,
            "summary": summary,
            "status": status,
            "limitations": "Human judgments and hardware timings require original evidence; quality metrics are not interchangeable. P01 primary score uses the declared exact saved-text semantic extraction; acceptance also requires the unchanged original strict JSON flag.",
        },
    )
    fields = [
        "job_id",
        "pair",
        "side",
        "input_id",
        "source_group",
        "seed",
        "status",
        "primary_score",
        "accepted",
        "generation_seconds",
        "end_to_end_seconds",
        "unchanged_region_mae",
        "review_seconds",
        "cost",
    ]
    fields += [
        "structured_parse_success",
        "semantic_answer",
        "semantic_value_available",
        "semantic_extraction_method",
        "semantic_source_field",
        "semantic_source_sha256",
        "semantic_rejection_reason",
        "semantic_analysis_policy",
        "structured_format_failure",
        "possibly_truncated",
        "failure_reason",
        "output_path",
        "task_success",
        "trace_integrity",
        "tool_errors",
        "agent_turns",
        "tool_calls",
        "output_tokens",
        "agent_termination",
        "asset_bytes",
        "asset_faces",
        "asset_textured",
    ]
    check_fields = set()
    for case in cases.values():
        for check in case["scoring"]["checks"]:
            check_fields.add(check)
    fields += sorted(check_fields)
    extra_fields = []
    for field in MANUAL_EXTRA:
        if field not in fields:
            extra_fields.append(field)
    fields += extra_fields
    write_csv(folder / "job_scores.csv", rows, fields)
    write_csv(folder / "pair_summary.csv", summary, list(summary[0]))
    rebuild_resource_table(root)
    valid_rows = []
    for row in rows:
        if row["status"] == "success":
            valid_rows.append(row)
    grouped = group_means(valid_rows)
    write_csv(
        folder / "source_group_scores.csv",
        grouped,
        ["pair", "side", "source_group", "primary_score", "observations"],
    )
    valid_count = 0
    for row in rows:
        if row["status"] == "success":
            valid_count += 1
    return {"summary": summary, "valid_outputs": valid_count}


# Draw comparisons only when actual reviewed scores are available.
# An empty experiment should not produce a chart that looks like measured results.
def plot_results(root):
    """Read saved summaries under root and return paths of generated charts."""
    root = Path(root)
    results = read_json(root / "results/results.json")
    has_scores = False
    pairs = set()
    for row in results["summary"]:
        pairs.add(row["pair"])
        if row["mean_valid_source_score"] is not None:
            has_scores = True
    if not has_scores:
        return {"figures": [], "status": "no measured quality scores"}
    # Keep the disposable font cache outside the submitted source folder.
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comp8410_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []
    for pair in sorted(pairs):
        values = []
        for row in results["summary"]:
            if row["pair"] == pair and row["mean_valid_source_score"] is not None:
                values.append(row)
        if not values:
            continue
        sides = []
        scores = []
        for row in values:
            sides.append(row["side"])
            scores.append(row["mean_valid_source_score"])
        figure, axis = plt.subplots(figsize=(5.5, 3.5))
        axis.bar(sides, scores, color=["#245b78", "#b16a2b"][: len(values)])
        axis.set(xlabel="System side", ylabel="Source-group mean score", title=pair)
        axis.set_ylim(0, 1)
        figure.tight_layout()
        path = root / f"results/{pair}_primary.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(str(path.relative_to(root)))
    status = "no measured quality scores"
    if created:
        status = "measured data only"
    return {"figures": created, "status": status}
