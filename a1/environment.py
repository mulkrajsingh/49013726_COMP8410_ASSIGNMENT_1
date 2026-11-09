# Read the active Python environment and available hardware; save a run record.
# Record installed package versions and device information for reproduction.
# This check does not install dependencies or demonstrate successful inference.

"""Record the tested interpreter and hardware without collecting credentials."""

import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

from .io import atomic_json, utc_now


# Read the installed packages, Python version and available GPU details.
# The record describes the environment actually used rather than the intended recipe.
def inspect_environment():
    """Return installed package, interpreter and available GPU information."""
    venv_config = Path(sys.prefix) / "pyvenv.cfg"
    include_system_packages = None
    if venv_config.is_file():
        for line in venv_config.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().lower() == "include-system-site-packages":
                if value.strip().lower() in {"true", "false"}:
                    include_system_packages = value.strip().lower() == "true"
    # Pillow and Pillow-SIMD can both have metadata. Report the loaded module too.
    pillow = sys.modules.get("PIL")
    result = {
        "recorded_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "pyvenv_config": str(venv_config) if venv_config.is_file() else None,
        "include_system_site_packages": include_system_packages,
        "pillow_imported_version": getattr(pillow, "__version__", None),
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    packages, origins, seen_names = {}, {}, set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        if canonical_name in seen_names:
            continue
        # Metadata follows import search order: keep the overlay before the base.
        seen_names.add(canonical_name)
        packages[name] = distribution.version
        direct = distribution.read_text("direct_url.json")
        if direct:
            source = json.loads(direct)
            url = urlsplit(source.get("url", ""))
            # Package source provenance is useful; credentials and URL queries are not.
            hostname = ""
            if url.scheme != "file":
                hostname = url.hostname or ""
            source["url"] = urlunsplit((url.scheme, hostname, url.path, "", ""))
            origins[name] = source
    result["packages"] = packages
    result["gpu"] = None
    result["package_source_origins"] = origins
    if importlib.util.find_spec("torch"):
        import torch

        result["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
    if shutil.which("nvidia-smi"):
        result["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    return result


# Save the measured environment and its installed package versions.
# A tested environment can then be recreated on another worker.
def capture_environment(root, worker="analysis", record=None):
    # Reuse the actual worker hardware record.
    """Save a worker environment record under root and return its relative path."""
    if record is None:
        record = inspect_environment()
    output = Path(root) / f"evidence/environments/{worker}.json"
    atomic_json(output, record)
    # Only packages actually present in this interpreter are called a lock.
    lock = output.with_suffix(".requirements.txt")
    package_lines = []
    for name, version in sorted(record["packages"].items()):
        package_lines.append(f"{name}=={version}")
    lock.write_text("\n".join(package_lines) + "\n")
    return str(output.relative_to(root))


# Check that the selected GPU supports the model settings.
# The worker should fail clearly if CUDA or the requested data type is unavailable.
def gpu_required(model):
    """Check the hardware for model and return the observed environment."""
    environment = inspect_environment()
    if not environment["gpu"]:
        raise RuntimeError(
            "This configuration requires an NVIDIA CUDA worker; the CPU inspection route remains available"
        )
    if (
        (model["parameters"].get("dtype") == "bfloat16"
         or model["parameters"].get("flow_dtype") == "bfloat16")
        and not environment["gpu"]["bf16_supported"]
    ):
        raise RuntimeError("The selected BF16 configuration is unsupported on this GPU")
    return environment
