# Read the fixed ABO selection and preparation settings; save attributed inputs.
# Keep source photographs, derived images and preparation records together.
# The pixel-based mask is a candidate that still needs the declared review.

# Prepare the already selected licensed product photographs.
# Keeping the originals and preparation records makes the inputs traceable.

"""Fetch a small attributed ABO subset, keeping originals and transformations."""

import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps

from .io import atomic_json, read_json, sha256, utc_now

ABO = "https://amazon-berkeley-objects.s3.amazonaws.com/"


# Find near white background pixels connected to the image border.
# This avoids removing white areas inside a product when making an edit mask.
def background_candidate(pixels):
    """Return a candidate background mask from RGB pixel values."""
    from scipy.ndimage import binary_propagation

    # Near white border pixels are only a candidate background.
    white = np.asarray(pixels).min(axis=-1) >= 245
    border = np.zeros_like(white)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return binary_propagation(border & white, mask=white)


# Download a bounded file through a temporary path.
# A failed download should not leave a partial file with the final input name.
def download(url, destination, max_bytes=128 * 1024 * 1024):
    """Fetch url into destination and return its URL, path, hash and size."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return {
            "url": url,
            "path": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    temporary = destination.with_name(destination.name + ".download")
    size = 0
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "COMP8410-student-evaluation/0.1"}),
            timeout=60,
        ) as response:
            expected = int(response.headers.get("Content-Length", 0))
            if expected > max_bytes:
                raise ValueError(f"Download exceeds its byte limit: {url}")
            with temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("The download exceeded its byte limit")
                    stream.write(chunk)
            if expected and size != expected:
                raise ValueError("Incomplete download")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "url": url,
        "path": str(destination),
        "sha256": sha256(destination),
        "bytes": size,
    }


# Prepare the shared product photos, single object views and edit masks.
# Both models in a pair should receive the same saved input bytes.
def prepare_abo(root):
    """Prepare the fixed ABO selection under root and return a count summary."""
    root = Path(root).resolve()
    # Reuse the saved selection instead of choosing new inputs after seeing results.
    selection = read_json(root / "data/abo_selection.json")
    provenance = []
    for product in selection["products"]:
        slot = product["slot"]
        row = product["image"]
        extension = Path(row["path"]).suffix
        url = ABO + "images/original/" + row["path"]
        raw = root / "data/raw/abo/images" / (slot + extension)
        record = download(url, raw, max_bytes=20 * 1024 * 1024)
        prepared = root / f"data/prepared/{slot}.png"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw) as image:
            # Pad without stretching. The unchanged original remains in raw storage.
            ImageOps.pad(image.convert("RGB"), (1024, 1024), color="white").save(
                prepared
            )
        record.update(
            {
                "path": str(raw.relative_to(root)),
                "slot": slot,
                "prepared": str(prepared.relative_to(root)),
                "prepared_sha256": sha256(prepared),
                "transformation": "Pillow ImageOps.pad RGB 1024x1024 with white fill",
            }
        )
        provenance.append(record)
        if slot in {"product01", "product02", "product03", "pilot_product"}:
            # The 3D input uses one source view rather than a source video.
            spin = product["spin_frames"][0]
            source = root / f"data/raw/abo/spins/{slot}/00.jpg"
            single_record = download(
                ABO + "spins/original/" + spin["path"], source, 20 * 1024 * 1024
            )
            single_record.update(
                path=str(source.relative_to(root)),
                slot=slot,
                azimuth_index=int(spin["azimuth"]),
            )
            provenance.append(single_record)
            single = root / f"data/prepared/{slot}_single.png"
            with Image.open(source) as image:
                ImageOps.pad(image.convert("RGB"), (1024, 1024), color="white").save(
                    single
                )
            provenance.append(
                {
                    "path": str(single.relative_to(root)),
                    "sha256": sha256(single),
                    "transformation": "first spin photograph padded to RGB 1024x1024; one object for image-to-3D",
                }
            )
    for slot in ["product01", "product02", "product03", "pilot_product"]:
        path = root / f"data/prepared/{slot}.png"
        pixels = np.asarray(Image.open(path).convert("RGB"))
        # A foreground preservation mask for background edits; review before final lock.
        Image.fromarray((~background_candidate(pixels)).astype(np.uint8) * 255).save(
            root / f"data/prepared/{slot}_foreground.png"
        )
    atomic_json(
        root / "data/provenance.json", {"created_at": utc_now(), "records": provenance}
    )
    return {
        "products": len(selection["products"]),
        "source_files": len(provenance),
        "labels": "pending human review",
    }


# Check that every input referenced by the current experiment exists.
# Missing source files should be found before starting a model run.
def validate_data(root, pair=None):
    """Return file-existence and checksum checks for the study under root."""
    checks = []
    for case in read_json(Path(root) / "config/study.json")["cases"]:
        if pair and case["pair"] != pair:
            continue
        for item in case["items"]:
            for path in item["files"]:
                actual = Path(root) / path
                present = actual.is_file()
                checksum = None
                if actual.is_file():
                    checksum = sha256(actual)
                checks.append(
                    {
                        "pair": case["pair"],
                        "input_id": item["id"],
                        "path": path,
                        "present": present,
                        "sha256": checksum,
                    }
                )
    return checks
