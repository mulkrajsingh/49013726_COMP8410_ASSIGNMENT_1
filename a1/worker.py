# Read one model's planned jobs; inspect them or save complete attempt records.
# Keep settings, raw outputs, validation, timing and errors for each attempt.
# Only the execution path calls the selected third-party model adapter.

"""One worker, one model, one append-only attempt log."""

import os
import sys
import time
import traceback
from pathlib import Path

from .backup import backup_attempt, backup_location, backup_project
from .environment import capture_environment, gpu_required
from .evidence import verified_file, verify_inputs
from .io import (
    append_attempt,
    atomic_json,
    file_records,
    project_path,
    read_attempts,
    read_json,
    sha256,
    utc_now,
)
from .media import validate_output
from .resources import ResourceMonitor
from .study import planned_jobs


def check_reproduction_environment(root, worker):
    """Require this worker's recorded new venv before a reproduction run."""
    path = os.environ.get("A1_REPRODUCTION_RECORD")
    if not path:
        raise ValueError("Set A1_REPRODUCTION_RECORD to the fresh-venv creation record")
    path = Path(path).expanduser().resolve()
    folder = Path(root).resolve() / "evidence/reproduction"
    if path.parent != folder or not path.is_file():
        raise ValueError("Use a creation record saved in evidence/reproduction")
    record = read_json(path)
    if (record.get("worker") != worker or
            record.get("target_absent_before_creation") is not True or
            record.get("creation_exit_code") != 0 or
            record.get("include_system_site_packages") is not False or
            record.get("empty_environment", {}).get("installed_distributions") != []):
        raise ValueError("The record does not establish this worker's fresh empty venv")
    target = Path(record["target_path"]).resolve()
    if Path(sys.prefix).resolve() != target or sys.prefix == sys.base_prefix:
        raise ValueError("Run reproduction using the new venv's own Python")
    config = target / "pyvenv.cfg"
    if (not config.is_file() or sha256(config) != record.get("pyvenv_config_sha256") or
            "include-system-site-packages = false" not in config.read_text().lower()):
        raise ValueError("The fresh venv configuration changed or allows system packages")
    baseline = read_json(Path(root) / f"evidence/environments/{worker}.json")
    baseline_env = Path(baseline["executable"]).parent.parent.resolve()
    if target == baseline_env or str(baseline_env) != record.get("baseline_environment_path"):
        raise ValueError("The reproduction venv must differ from the baseline worker environment")
    return path.stem


# Keep pilots, final runs and reproductions in separate logs.
# Both the scorer and the worker use this same location.
def log_path(root, job):
    """Return the attempt-log path for job under project root."""
    return (
        Path(root)
        / f"outputs/{job['pair']}/{job['side']}/{job['phase']}/attempts.jsonl"
    )


# Recheck an earlier successful attempt before reusing it.
# Changed inputs or damaged files must not count as completed work.
def successful_record(root, job):
    """Return a reusable successful attempt for job, or None."""
    records = read_attempts(log_path(root, job))
    for row in reversed(records):
        if row["job_id"] != job["job_id"]:
            continue
        if row["status"] != "success":
            return None
        try:
            verify_inputs(root, job, row)
            for artifact in row["artifacts"]:
                verified_file(root, artifact)
            validate_output(project_path(root, row["output_path"]), job["output"])
        except (OSError, ValueError, KeyError):
            # Corrupt saved files need a new attempt, not a false success.
            return None
        return row
    return None


# Run one selected model, or inspect its jobs without executing them.
# Injected runners are only used by software tests in temporary projects.
def run_worker(
    root, pair, side, phase="pilot", execute=False, runner=None, backup_root=None,
    monitor_resources=None,
):
    """Inspect or execute one pair/side/phase of the study under root."""
    root = Path(root).resolve()
    jobs = []
    for job in planned_jobs(root, pair, phase):
        if job["side"] == side:
            jobs.append(job)
    if not jobs:
        raise ValueError(
            "Choose an active pair/side worker from the study configuration"
        )
    # Inspection stops here, before loading a model or starting an attempt.
    if not execute:
        return {
            "mode": "inspection",
            "jobs": len(jobs),
            "worker": f"{pair}_{side}",
            "phase": phase,
            "will_download_or_generate": False,
        }

    model = read_json(root / "config/models.json")[f"{pair}_{side}"]
    real_run = runner is None
    if monitor_resources is None:
        monitor_resources = real_run
    if real_run:
        reproduction_name = None
        if phase == "reproduction":
            reproduction_name = check_reproduction_environment(root, f"{pair}_{side}")
        # A marker may use any suitable Linux CUDA machine. Select a normal
        # persistent folder; the historical RunPod/Mac collector is optional.
        persistent = Path(os.environ.get("A1_PERSISTENT_ROOT", root.parent)).expanduser().resolve()
        persistent.mkdir(parents=True, exist_ok=True)
        if backup_root is not None:
            backup_root = backup_location(root, backup_root)
        if phase == "final":
            study = read_json(root / "config/study.json")
            case = next(c for c in study["cases"] if c["pair"] == pair)
            labels = read_json(root / case["annotations_file"])
            unreviewed = sorted({j["input_id"] for j in jobs if not labels.get(j["input_id"], {}).get("reviewed")})
            if unreviewed:
                raise ValueError(f"Review input labels/masks before final runs: {unreviewed}")
        # Keep runtime compilations after a container restart too.
        for variable, name in [("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                               ("TRITON_CACHE_DIR", "triton"),
                               ("CUDA_CACHE_PATH", "cuda")]:
            cache = persistent / "cache" / name / model["worker"]
            cache.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(cache)
        import torch
        from .models import ModelRunner

        backup_root = backup_location(root, backup_root)
        hardware = gpu_required(model)
        environment_name = f"{pair}_{side}"
        if reproduction_name is not None:
            environment_name = f"reproduction_{reproduction_name}"
        capture_environment(root, environment_name, record=hardware)
        runner = ModelRunner(root, model)
    else:
        hardware = {"software_test": True}
        # Tests use a backup only when they explicitly provide one.
        if backup_root is not None:
            backup_root = backup_location(root, backup_root)

    if backup_root is not None:
        # Save source inputs and code before loading a model.
        backup_project(root, backup_root)
    results = []
    loaded = False

    # Copy each attempt before moving to the next job.
    # If copying fails, the valid output remains available for a copy retry.
    def preserve(job, record):
        """Back up job and record, and append its backup receipt."""
        if backup_root is None:
            return
        backup_log = log_path(root, job).with_name("backups.jsonl")
        try:
            receipt = backup_attempt(root, job, record, backup_root)
        except Exception as error:
            append_attempt(
                backup_log,
                {
                    "job_id": job["job_id"],
                    "attempt": record["attempt"],
                    "status": "backup_failed",
                    "at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        append_attempt(
            backup_log,
            {
                "job_id": job["job_id"],
                "attempt": record["attempt"],
                "status": "backed_up",
                **receipt,
            },
        )

    for job in jobs:
        existing = successful_record(root, job)
        if existing:
            preserve(job, existing)
            results.append(
                {
                    "job_id": job["job_id"],
                    "status": "resumed",
                    "attempt": existing["attempt"],
                }
            )
            continue
        previous = []
        for row in read_attempts(log_path(root, job)):
            if row["job_id"] == job["job_id"]:
                previous.append(row)
        if previous:
            last = previous[-1]
            if last["status"] == "started":
                # A process crash may have left only the start record and partial files.
                last = {
                    **last,
                    "status": "interrupted",
                    "error": "Previous process ended before saving its finish record",
                }
                append_attempt(log_path(root, job), last)
            elif last["status"] == "success":
                # The resume check above found changed inputs or damaged artifacts.
                last = {
                    **last,
                    "status": "invalidated",
                    "error": "Saved artifacts or input bytes changed",
                }
                append_attempt(log_path(root, job), last)
            # Retry a failed copy before creating another model attempt.
            preserve(job, last)
        previous_attempts = []
        for row in previous:
            previous_attempts.append(row["attempt"])
        attempt = max(previous_attempts, default=0) + 1
        folder = log_path(root, job).parent / job["job_id"] / f"attempt_{attempt:02d}"
        temporary = folder.with_name(folder.name + ".partial")
        # Existing partial files are never overwritten.
        temporary.mkdir(parents=True)
        timing_boundary = "software test"
        if real_run:
            timing_boundary = runner.timing_boundary
        record = {
            "job_id": job["job_id"],
            "attempt": attempt,
            "pair": pair,
            "side": side,
            "phase": phase,
            "source_group": job["source_group"],
            "input_id": job["input_id"],
            "seed": job["seed"],
            "started_at": utc_now(),
            "status": "started",
            "model": job["model"],
            "hardware": hardware,
            "generation_timing_boundary": timing_boundary,
            "input_files": file_records(root, job["item"]["files"]),
            "cost": None,
            "cost_source": "unavailable until actual billing is supplied",
            "resource_directory": str((temporary / "resources").relative_to(root)) if monitor_resources else None,
        }
        append_attempt(log_path(root, job), record)
        started = time.perf_counter()
        monitor = None
        peak_stats_reset = False

        def finish_resources():
            # Attempt this on both success and failure, without hiding the model error.
            if monitor is not None:
                try:
                    record["resources"] = monitor.stop()
                except Exception as error:
                    record["resources"] = {"complete": False, "errors": [str(error)]}
            if real_run and peak_stats_reset:
                try:
                    record["torch_peaks_by_device"] = [
                        {"device": device,
                         "allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "reserved_bytes": torch.cuda.max_memory_reserved(device)}
                        for device in range(torch.cuda.device_count())
                    ]
                    record["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    record["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    record["gpu_memory_boundary"] = (
                        "PyTorch allocator per device in this process; native allocations excluded; "
                        "resources/gpu_peaks separately contains sampled whole-device usage"
                    )
                except Exception as error:
                    record["torch_memory_error"] = f"{type(error).__name__}: {error}"

        try:
            case = None
            for candidate in read_json(root / "config/study.json")["cases"]:
                if candidate["pair"] == pair:
                    case = candidate
                    break
            if case is None:
                raise StopIteration
            atomic_json(
                temporary / "settings.json",
                {
                    "job": job,
                    "model": model,
                    "annotations": read_json(root / case["annotations_file"]),
                    "input_files": record["input_files"],
                    "environment": hardware,
                },
            )
            if monitor_resources:
                paths = [root, root.anchor, backup_root, "/workspace",
                         os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
                         os.environ.get("TMPDIR", "/tmp")]
                monitor = ResourceMonitor(
                    temporary / "resources", paths=[p for p in paths if p],
                    require_gpu=real_run,
                )
                monitor.start()
                monitor.stage("model_load" if not loaded else "model_reused")
            if real_run:
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                peak_stats_reset = True
            cold_start = time.perf_counter()
            record["model_reused"] = loaded
            if not loaded:
                runner.load()
                loaded = True
            if real_run:
                # Wait for model loading before stopping its timer.
                torch.cuda.synchronize()
            record["model_load_seconds"] = 0.0 if record["model_reused"] else time.perf_counter() - cold_start
            if monitor is not None:
                monitor.stage("generation_export_preview")
            infer_start = time.perf_counter()
            relative_output = runner.generate(job, temporary)
            if (
                Path(relative_output).is_absolute()
                or ".." in Path(relative_output).parts
            ):
                raise ValueError(
                    "A runner output must stay inside its attempt directory"
                )
            if real_run:
                # Queued GPU work belongs inside the generation time.
                torch.cuda.synchronize()
            record["generation_seconds"] = time.perf_counter() - infer_start
            if monitor is not None:
                monitor.stage("output_validation")
            # Reopening checks the file contract, not whether a person accepts the result.
            record["output_validation"] = validate_output(
                temporary / relative_output, job["output"]
            )
            finish_resources()
            # Include materials and previews so a resume checks the complete workflow.
            artifacts = []
            for file in sorted(temporary.rglob("*")):
                if file.is_file():
                    artifacts.append(
                        {
                            "path": str(
                                (folder / file.relative_to(temporary)).relative_to(root)
                            ),
                            "sha256": sha256(file),
                            "bytes": file.stat().st_size,
                        }
                    )
            os.replace(temporary, folder)
            record.update(
                status="success",
                output_path=str((folder / relative_output).relative_to(root)),
                artifacts=artifacts,
            )
            if monitor is not None:
                record["resource_directory"] = str((folder / "resources").relative_to(root))
        except BaseException as error:
            finish_resources()
            atomic_json(temporary / "failure.json", {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            # Preserve errors and user interruptions, then stop this configuration.
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            failure_status = "failed"
            if interrupted:
                failure_status = "interrupted"
            record.update(
                status=failure_status,
                error=f"{type(error).__name__}: {error}",
                partial_directory=str(temporary.relative_to(root)),
            )
            record["finished_at"] = utc_now()
            record["end_to_end_seconds"] = time.perf_counter() - started
            append_attempt(log_path(root, job), record)
            try:
                preserve(job, record)
            except Exception as backup_error:
                error.add_note(f"Backup also failed: {backup_error}")
            raise
        record["finished_at"] = utc_now()
        record["end_to_end_seconds"] = time.perf_counter() - started
        append_attempt(log_path(root, job), record)
        preserve(job, record)
        results.append(
            {"job_id": job["job_id"], "status": record["status"], "attempt": attempt}
        )
    return results
