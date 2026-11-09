# Read answers, labels, image arrays or review flags; return task measures.
# Keep missing reviews separate from failed checks.
# Group repeated seeds and questions by their source product before averaging.

"""Direct metrics with declared ranges and no model judge dependency."""

import math
import re
from collections import defaultdict

import numpy as np


def normalized_answer(answer):
    """Return answer as lowercase text with punctuation removed."""
    return re.sub(r"[^\w\s]", "", str(answer).casefold()).strip()


# Compare an answer with the reviewed accepted answers.
# We only score known facts automatically when the labels have been checked.
def supported_answer(answer, label):
    """Return 1.0 for a reviewed accepted answer, or 0.0 for a mismatch."""
    if not label.get("reviewed") or not label.get("accepted_answers"):
        raise ValueError("Ground truth must be reviewed before scoring")
    accepted = set()
    for value in label["accepted_answers"]:
        accepted.add(normalized_answer(value))
    answer_matches = normalized_answer(answer) in accepted
    return float(answer_matches)


# Compare fixed pixel positions inside the mask; this does not track a moved product.
def masked_mae(reference, prediction, unchanged_mask):
    """Return the mean absolute pixel error inside unchanged_mask."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    mask = np.asarray(unchanged_mask, dtype=bool)
    if reference.shape != prediction.shape or mask.shape != reference.shape[:2]:
        raise ValueError("Preservation MAE needs a fixed shared pixel alignment")
    if not mask.any():
        raise ValueError("The unchanged region is empty")
    return float(np.abs(reference - prediction)[mask].mean())


# Keep the score missing until every required review check has a value.
def checklist_score(values):
    """Return the fraction of passed checks, or None if a review is missing."""
    if not values:
        return None
    for value in values:
        if value is not True and value is not False:
            return None
    return sum(values) / len(values)


# Average repeated questions or seeds within their source product first.
# This avoids giving a product extra weight just because it has more outputs.
def group_means(rows, key="primary_score"):
    """Return one average per experiment, model side and source product."""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(value):
            group = (row["pair"], row["side"], row["source_group"])
            groups[group].append(value)

    averages = []
    for group, values in sorted(groups.items()):
        pair, side, source = group
        averages.append(
            {
                "pair": pair,
                "side": side,
                "source_group": source,
                key: sum(values) / len(values),
                "observations": len(values),
            }
        )
    return averages


# Count planned, valid, failed and reviewed jobs for each model.
# Failed attempts stay in the planned total instead of disappearing from the comparison.
def summarize(planned, rows):
    """Return completion and quality summaries for each experiment/model."""
    index = {}
    for row in rows:
        index[row["job_id"]] = row
    if len(index) != len(rows):
        raise ValueError("Summary rows must contain one resolved row per planned job")
    planned_ids = set()
    model_sides = set()
    for job in planned:
        planned_ids.add(job["job_id"])
        model_sides.add((job["pair"], job["side"]))
    if set(index) - planned_ids:
        raise ValueError("Results include jobs outside the planned manifest")

    summary = []
    for pair, side in sorted(model_sides):
        jobs = []
        results = []
        for job in planned:
            if job["pair"] == pair and job["side"] == side:
                jobs.append(job)
                if job["job_id"] in index:
                    results.append(index[job["job_id"]])

        valid = []
        accepted = []
        judged = []
        quality = []
        failed = 0
        for result in results:
            if result.get("status") in {"failed", "interrupted", "invalidated"}:
                failed += 1
            if result.get("status") == "success":
                valid.append(result)
                if result.get("accepted") is True:
                    accepted.append(result)
                if result.get("accepted") is not None:
                    judged.append(result)
                if result.get("primary_score") is not None:
                    quality.append(result)

        # Average each product first so repeated seeds do not add product weight.
        grouped = group_means(quality)
        complete = len(valid) + failed == len(jobs)
        acceptance_rate = None
        if complete and len(judged) == len(valid):
            acceptance_rate = len(accepted) / len(jobs)
        mean_score = None
        if grouped:
            group_scores = []
            for group in grouped:
                group_scores.append(group["primary_score"])
            mean_score = sum(group_scores) / len(grouped)
        summary.append(
            {
                "pair": pair,
                "side": side,
                "planned": len(jobs),
                "valid": len(valid),
                "failed": failed,
                "pending": len(jobs) - len(valid) - failed,
                "judged_valid": len(judged),
                "accepted": len(accepted),
                "completion_rate": len(valid) / len(jobs),
                "acceptance_rate_all_planned": acceptance_rate,
                "source_groups_scored": len(grouped),
                "mean_valid_source_score": mean_score,
            }
        )
    return summary
