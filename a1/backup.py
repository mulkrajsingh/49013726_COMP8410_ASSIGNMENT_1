# Copy selected code, inputs and attempt files to a separate backup folder.
# Return a receipt only after checking the copied files against their checksums.
# These copies preserve evidence; they do not add experiment results.

"""Verified snapshots of project files, attempts and analysis evidence."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from .io import atomic_json, fingerprint, project_path, read_json, sha256, utc_now


def wait_for_local_copy(folder, destination):
    """Wait for the Mac to download and verify this snapshot."""
    if os.environ.get("A1_REQUIRE_LOCAL_COPY") != "1":
        return None
    manifest = verify_backup(folder)
    manifest_hash = sha256(folder / "backup.json")
    atomic_json(folder / "local_copy_request.json", {
        "requested_at": utc_now(), "manifest_sha256": manifest_hash,
        "files": len(manifest["files"]),
        "bytes": sum(row["bytes"] for row in manifest["files"]),
    })
    timeout = float(os.environ.get("A1_LOCAL_COPY_TIMEOUT_SECONDS", "180"))
    if not 0 < timeout <= 3600:
        raise ValueError("Local-copy timeout must be between 0 and 3600 seconds")
    deadline = time.monotonic() + timeout
    receipt_file = folder / "local_copy_receipt.json"
    while True:
        if receipt_file.is_file():
            receipt = read_json(receipt_file)
            if (receipt.get("status") != "verified"
                or receipt.get("manifest_sha256") != manifest_hash
                or receipt.get("files_verified") != len(manifest["files"])):
                raise ValueError("The local-copy receipt does not match this snapshot")
            return receipt
        if time.monotonic() >= deadline:
            raise TimeoutError("Mac backup not confirmed; batch stopped. Keep the network volume and restart the local collector.")
        time.sleep(min(1, max(0, deadline - time.monotonic())))


# Read and check the explicitly selected backup folder.
# A folder inside the project would be lost with the original workspace.
def backup_location(root, selected):
    """Return selected as an absolute backup folder outside root."""
    if selected is None:
        raise ValueError(
            "Set backup_root to a durable folder before executing or backing up"
        )
    destination = Path(selected).expanduser().resolve()
    if destination.is_relative_to(Path(root).resolve()):
        raise ValueError("The backup directory must be outside the project")
    return destination


# Reopen the saved manifest and compare every copied file.
# An existing but damaged backup must not be treated as successful.
def verify_backup(folder):
    """Read backup.json in folder and return it after checking all files."""
    manifest = read_json(Path(folder) / "backup.json")
    for item in manifest["files"]:
        actual = project_path(folder, item["path"])
        if not actual.is_file() or sha256(actual) != item["sha256"]:
            raise ValueError(f"Backup content differs: {item['path']}")
    return manifest


# Copy the requested files into an immutable snapshot.
# Identical files reuse their verified snapshot instead of making another copy.
def backup_files(root, destination, name, paths):
    """Copy paths from root to a verified snapshot and return its receipt."""
    root = Path(root).resolve()
    destination = backup_location(root, destination)
    files = []
    for path in sorted(set(paths)):
        path = Path(path).resolve()
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                    "bytes": Path(path).stat().st_size,
                }
            )
    identity = fingerprint(files)
    folder = destination / name / identity[:16]
    if folder.exists():
        verify_backup(folder)
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=folder.parent))
        try:
            for item in files:
                target = temporary / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_path(root, item["path"]), target)
            atomic_json(
                temporary / "backup.json", {"created_at": utc_now(), "files": files}
            )
            verify_backup(temporary)
            os.replace(temporary, folder)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    local_copy = wait_for_local_copy(folder, destination)
    return {"path": str(folder), "verified_at": utc_now(), "local_copy": local_copy}


# Save code, selected data, settings, environments, reviews and result tables.
# Run this at worker startup and after scoring or self evaluation.
def backup_project(root, destination):
    """Copy the project's reproducibility files to destination and return a receipt."""
    root = Path(root).resolve()
    paths = []
    folders = [
        "a1",
        "config",
        "notebooks",
        "references",
        "scripts",
        "evidence",
        "results",
        "environments",
        "data",
    ]
    for name in folders:
        for path in (root / name).rglob("*"):
            excluded = False
            for part in path.parts:
                if part in {"__pycache__", ".DS_Store"}:
                    excluded = True
                    break
            if path.is_file() and not excluded:
                if path.is_symlink():
                    raise ValueError(f"Do not archive linked data or credentials: {path}")
                paths.append(path)
    for name in [
        "README.md",
        "NOTICE.md",
        "requirements-analysis.txt",
        "data/abo_selection.json",
        "data/provenance.json",
        "data/manual_reviews.csv",
    ]:
        paths.append(root / name)
    paths += list((root / "outputs").rglob("*.jsonl"))
    study = read_json(root / "config/study.json")
    for case in study["cases"]:
        paths.append(root / case["annotations_file"])
        for item in case["items"]:
            for relative in item["files"]:
                paths.append(root / relative)
    provenance = root / "data/provenance.json"
    if provenance.exists():
        for record in read_json(provenance)["records"]:
            paths.append(root / record["path"])
            if record.get("prepared"):
                paths.append(root / record["prepared"])
        # Keep source notices as well as the photographs they describe.
        for record in read_json(root / "data/abo_selection.json")["metadata"]:
            if Path(record["path"]).suffix.lower() in {".md", ".txt"}:
                paths.append(root / record["path"])
    return backup_files(root, destination, "project", paths)


# Preserve one attempt and the worker log, including partial failed output.
# All attempt folders remain separate so retries never erase earlier evidence.
def backup_attempt(root, job, record, destination):
    """Back up one recorded job attempt and return the copy receipt."""
    root = Path(root).resolve()
    log = root / f"outputs/{job['pair']}/{job['side']}/{job['phase']}/attempts.jsonl"
    parent = log.parent / job["job_id"]
    paths = [log]
    for path in parent.glob(f"attempt_{record['attempt']:02d}*/**/*"):
        if path.is_file():
            paths.append(path)
    return backup_files(
        root,
        destination,
        f"attempts/{job['pair']}/{job['side']}/{job['phase']}/{job['job_id']}/attempt_{record['attempt']:02d}",
        paths,
    )
