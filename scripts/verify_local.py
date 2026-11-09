# Read source provenance and active prepared images; verify hashes and sizes.
# Write a verification receipt in the system temporary directory.
# This checks local input files, not model execution or human review.

"""Verify active data bytes and image contracts without running models."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

from pathlib import Path
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.data import validate_data
from a1.io import atomic_json, project_path, read_json, sha256, utc_now
from a1.media import validate_output


# Check source checksums and prepared input image contracts.
# The saved receipt records data checks without claiming model execution.
def main():
    """Verify supplied source bytes and active image contracts under ROOT.

    Read saved provenance, check hashes and inspect prepared image/mask sizes.
    Write a receipt in the system temporary directory and print a count summary.
    Return None. This uses local inputs only and does not run models or
    complete the student label/mask review."""
    verified = {}
    for record in read_json(ROOT / "data/provenance.json")["records"]:
        references = [(record["path"], record["sha256"])]
        if record.get("prepared_sha256"):
            references.append((record["prepared"], record["prepared_sha256"]))
        for relative, expected in references:
            path = project_path(ROOT, relative)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Source/provenance checksum differs: {relative}")
            verified[relative] = expected
    # Check only the files referenced by the active four experiments.
    checks = validate_data(ROOT)
    missing = []
    for r in checks:
        if not r["present"]:
            missing.append(r)
    if missing:
        raise ValueError(f"Missing active input files: {missing}")
    images = []
    image_paths = set()
    for check in checks:
        if check["path"].endswith(".png"):
            image_paths.add(check["path"])
    for relative in sorted(image_paths):
        if relative.endswith("_foreground.png"):
            with Image.open(ROOT / relative) as mask:
                mask.load()
                if mask.size != (1024, 1024):
                    raise ValueError("Edit mask size differs")
            images.append(
                {"path": relative, "kind": "foreground mask; file contract checked"}
            )
        else:
            images.append(
                {
                    "path": relative,
                    **validate_output(
                        ROOT / relative,
                        {"kind": "image", "width": 1024, "height": 1024},
                    ),
                }
            )
    result = {
        "verified_at": utc_now(),
        "source_and_derivative_hashes_verified": len(verified),
        "images": images,
        "required_file_references": len(checks),
        "missing_prepared_inputs": [],
        "scope": "active source bytes and image contracts only; this check does not perform human review, oracle evaluation or model execution",
    }
    atomic_json(Path(tempfile.gettempdir()) / "comp8410_checks/data_verification.json", result)
    print(
        f"Verified {len(verified)} source/derivative hashes and {len(images)} active images/masks; no models run"
    )


if __name__ == "__main__":
    main()
