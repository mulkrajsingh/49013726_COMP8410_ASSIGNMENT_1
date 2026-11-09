# Read saved P01/P04 outputs and resource tables; write descriptive diagnostics.
# Separate answer extraction, formatting, task success and resource measurements.
# These tables explain the recorded results without replacing their scores.

"""Explain the saved text-model results without changing the primary scores."""

import csv
from collections import Counter
from pathlib import Path
from statistics import median

from .agents import facts_equal, score_agent
from .io import project_path, read_json, write_csv
from .metrics import supported_answer
from .semantic_answers import semantic_answer
from .study import planned_jobs
from .worker import successful_record


# Keep format failures separate from factual failures. These are descriptive
# checks of saved responses, not a new parser or a replacement success measure.
def rebuild_diagnostics(root):
    root = Path(root)
    p01, p04 = [], []
    catalogue = read_json(root / "data/retail/catalogue.json")
    annotations = {pair: read_json(root / f"data/annotations/{pair}.json") for pair in ["P01", "P04"]}
    for job in planned_jobs(root):
        if job["pair"] not in annotations:
            continue
        record = successful_record(root, job)
        if record is None:
            continue
        saved = read_json(project_path(root, record["output_path"]))
        labels = annotations[job["pair"]][job["input_id"]]
        row = {key: job[key] for key in ["job_id", "pair", "side", "input_id", "source_group"]}
        if job["pair"] == "P01":
            extracted = semantic_answer(saved)
            score = supported_answer(extracted["semantic_answer"], labels) if extracted["semantic_value_available"] else 0.0
            question = {"q1": "colour", "q2": "retail_unit_count", "q3": "unobservable_weight"}[job["input_id"].split("_")[-1]]
            row.update(question_type=question, supported_answer_score=score,
                       strict_json=saved.get("structured_parse_success"),
                       extraction_method=extracted["semantic_extraction_method"],
                       extracted_answer=extracted["semantic_answer"], output_path=record["output_path"])
            p01.append(row)
            continue
        expected = labels["expected"]
        measures = score_agent(saved, catalogue, expected)
        final = saved.get("final")
        facts = final["facts"] if final else {}
        required = expected["facts"]
        present = all(key in facts for key in required)
        correct = present and facts_equal({key: facts[key] for key in required}, required)
        status_ok = final is not None and final["status"] == expected["status"]
        extra = sorted(set(facts) - set(required))
        if measures["task_success"]:
            reason = "strict_success"
        elif final is None:
            reason = "no_valid_final"
        elif not status_ok:
            reason = "status_mismatch"
        elif not correct:
            reason = "missing_or_incorrect_requested_facts"
        elif extra:
            reason = "extra_fact_fields"
        else:
            reason = "lookup_or_trace_requirement"
        row.update(strict_task_success=measures["task_success"], valid_final=final is not None,
                   expected_status=expected["status"], observed_status=final["status"] if final else "",
                   requested_facts_match=correct if final else False,
                   extra_fact_fields=";".join(extra), failure_category=reason,
                   termination=saved["termination"], tool_errors=measures["tool_errors"],
                   output_path=record["output_path"])
        p04.append(row)
    for name, rows in [("P01_diagnostics.csv", p01), ("P04_diagnostics.csv", p04)]:
        if rows:
            write_csv(root / "results" / name, rows, list(rows[0]))
    return {"P01_rows": len(p01), "P04_rows": len(p04),
            "P04_categories": {side: dict(Counter(row["failure_category"] for row in p04 if row["side"] == side)) for side in ["A", "B"]}}


# Summarise main runs only. Pilots, reproductions and failed setup attempts
# stay in resource_usage.csv, but must not enter the main-model timing table.
def rebuild_model_statistics(root):
    root = Path(root)
    with (root / "results/resource_usage.csv").open(newline="") as stream:
        resources = list(csv.DictReader(stream))
    with (root / "results/job_scores.csv").open(newline="") as stream:
        scores = list(csv.DictReader(stream))
    rows = []
    for pair in ["P01", "P02", "P03", "P04"]:
        for side in ["A", "B"]:
            runs = [r for r in resources if r["pair"] == pair and r["side"] == side and r["phase"] == "final" and r["status"] == "success"]
            if not runs:
                continue
            cold = [float(r["model_load_seconds"]) for r in runs if r["model_load_seconds"] and float(r["model_load_seconds"]) > 0]
            warm = [float(r["generation_seconds"]) for r in runs if r["model_reused"].lower() == "true"]
            row = {"pair": pair, "side": side, "final_runs": len(runs),
                   "cold_load_seconds": cold[0] if cold else None,
                   "median_generation_seconds": median(float(r["generation_seconds"]) for r in runs),
                   "warm_runs": len(warm), "warm_median_seconds": median(warm) if warm else None,
                   "median_end_to_end_seconds": median(float(r["end_to_end_seconds"]) for r in runs),
                   "observed_load_plus_generation_seconds": sum(float(r["model_load_seconds"] or 0) + float(r["generation_seconds"]) for r in runs)}
            for key in ["gpu_largest_device_peak_used_bytes", "process_peak_rss_bytes"]:
                values = [float(r[key]) for r in runs if r[key]]
                row[key] = max(values) if values else None
            group = [r for r in scores if r["pair"] == pair and r["side"] == side]
            judged = [r for r in group if r["accepted"] != ""]
            row["accepted"] = sum(r["accepted"].lower() == "true" for r in judged) if len(judged) == len(group) else None
            row["judged_outputs"] = len(judged)
            rows.append(row)
    write_csv(root / "results/model_statistics.csv", rows, list(rows[0]))
    return rows
