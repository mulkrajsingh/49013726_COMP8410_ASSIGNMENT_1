# Read the fixed study and model registry; return planned job records.
# Each job fixes its input, model side, phase and seed before execution.
# The plan supplies the denominator used to report incomplete or failed jobs.

"""Build pilot, final and reproduction jobs from the saved configuration."""

from pathlib import Path
from .io import fingerprint, read_json


# Each job records its exact model settings, prompt, seed and output contract.
# A changed scientific setting creates a different job rather than replacing an old run.
def planned_jobs(root, pair=None, phase="final"):
    """Read the saved study and return the requested list of model runs."""
    if phase not in {"pilot", "final", "reproduction"}:
        raise ValueError("Phase must be pilot, reproduction or final")
    study = read_json(Path(root) / "config/study.json")
    models = read_json(Path(root) / "config/models.json")
    jobs = []
    for case in study["cases"]:
        if pair and case["pair"] != pair:
            continue
        if phase == "reproduction" and case["pair"] not in study["reproduction_pairs"]:
            continue

        # Reproduction repeats the pilot input, but gets a separate job ID.
        input_split = phase
        if phase == "reproduction":
            input_split = "pilot"
        seeds = case["seeds"]
        if phase != "final":
            seeds = case["seeds"][:1]

        for side in ("A", "B"):
            model = models[f"{case['pair']}_{side}"]
            for item in case["items"]:
                if item["split"] != input_split:
                    continue
                for seed in seeds:
                    model_settings = {}
                    for key in [
                        "model_id",
                        "implementation",
                        "weight_revision",
                        "code_revision",
                        "components",
                        "parameters",
                    ]:
                        model_settings[key] = model[key]
                    identity = {
                        "pair": case["pair"],
                        "side": side,
                        "phase": phase,
                        "input_id": item["id"],
                        "source_group": item["source_group"],
                        "seed": seed,
                        "protocol_version": study["protocol_version"],
                        "model": model_settings,
                        "item": item,
                        "scoring": case["scoring"],
                        "output": case["output"],
                    }
                    # Include a settings checksum so changed settings get a new ID.
                    settings_hash = fingerprint(identity)[:16]
                    job_id = f"{case['pair']}_{side}_{phase}_{item['id']}_{seed}_{settings_hash}"
                    job = {"job_id": job_id}
                    job.update(identity)
                    jobs.append(job)
    return jobs
