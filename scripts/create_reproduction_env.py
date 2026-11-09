# Read a worker selection and an unused environment path.
# Create a separate Python environment and save its creation record.
# This prepares a reproduction environment; it does not reproduce model outputs.

"""Create an empty reproduction venv and record how it was created.

This does not install model packages or run a reproduction experiment.
"""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.io import read_json, sha256, utc_now

WORKERS = ["P01_A", "P01_B", "P03_A", "P03_B"]


def create_environment(worker, target, python, root=ROOT):
    """Create a new isolated venv using the explicitly supplied Python."""
    root = Path(root).resolve()
    target = Path(target).expanduser()
    if target.exists() or target.is_symlink():
        raise ValueError("Reproduction target already exists; choose a new path")
    target = target.resolve()
    python = Path(python).expanduser().absolute()
    if worker not in WORKERS or not python.is_file():
        raise ValueError("Choose a P01/P03 worker and an existing Python executable")
    baseline = root / f"evidence/environments/{worker}.json"
    baseline_record = read_json(baseline)
    baseline_python = Path(baseline_record["executable"])
    if not baseline_python.is_absolute():
        raise ValueError("The baseline environment needs its absolute Python path")
    baseline_env = baseline_python.parent.parent.resolve()
    if target == baseline_env:
        raise ValueError("Reproduction must use a different environment")
    record_path = root / "evidence/reproduction" / f"{worker}_{target.name}.json"
    if record_path.exists():
        raise ValueError("A reproduction creation record with this name already exists")
    # Reserve the new path so a concurrent creator cannot reuse it.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    command = [str(python), "-I", "-m", "venv", "--without-pip", str(target)]
    subprocess.run(command, check=True)
    probe = """import importlib.metadata,json,site,sys
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'executable':sys.executable,'python_version':sys.version,
 'site_packages':site.getsitepackages(),
 'installed_distributions':sorted(d.metadata['Name'] for d in importlib.metadata.distributions())}))
"""
    created = json.loads(subprocess.check_output(
        [str(target / "bin/python"), "-I", "-c", probe], text=True
    ))
    config = target / "pyvenv.cfg"
    isolated = "include-system-site-packages = false" in config.read_text().lower()
    if (Path(created["prefix"]).resolve() != target or
            created["prefix"] == created["base_prefix"] or
            created["installed_distributions"] or not isolated):
        raise ValueError("New venv is not empty and isolated; keep it for inspection")
    record = {
        "created_at_utc": utc_now(), "worker": worker,
        "target_path": str(target), "base_python": str(python),
        "baseline_environment_record": str(baseline.relative_to(root)),
        "baseline_environment_sha256": sha256(baseline),
        "baseline_environment_path": str(baseline_env),
        "target_absent_before_creation": True,
        "include_system_site_packages": False,
        "creation_command": command, "creation_exit_code": 0,
        "pyvenv_config_sha256": sha256(config),
        "empty_environment": created,
        "scope": "Fresh empty venv creation only; model installation and pipeline reproduction are not yet complete",
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS, required=True)
    parser.add_argument("--python", type=Path, required=True,
                        help="Explicit path to the Python used to create the venv")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(create_environment(args.worker, args.target, args.python))
