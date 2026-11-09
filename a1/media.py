# Reopen a saved output and check it against its declared file contract.
# Return measured file properties, or fail when the contract is not met.
# A valid file is separate from a satisfactory answer, edit or 3D draft.

"""Reopen every output; a filename alone never establishes success."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


# Load an image as RGB values between zero and one.
# The image metrics need the same colour format and value range.
def image_array(path):
    """Read the image at path and return RGB values between zero and one."""
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


# Read video frames together with their video metadata.
# The output checks need to inspect the whole clip rather than one thumbnail.
def video_frames(path):
    """Yield each decoded frame at path together with the video metadata."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            yield np.asarray(frame, dtype=np.float64) / 255.0, metadata
    finally:
        reader.close()


# Check the saved file against the output type required by the experiment.
# This rejects broken files before they are counted as successful outputs.
def validate_output(path, contract):
    """Check path against the experiment contract and return measured details."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty output: {path}")
    # Each experiment declares the type of output it must save.
    kind = contract["kind"]
    if kind == "agent":
        from .agents import validate_trace

        return validate_trace(json.loads(path.read_text()))
    if kind == "answer":
        result = json.loads(path.read_text())
        if not isinstance(result.get("answer"), str) or not result["answer"].strip():
            raise ValueError("An answer must contain nonempty text")
        return {"kind": kind, "characters": len(result["answer"])}
    if kind == "image":
        pixels = image_array(path)
        height, width = pixels.shape[:2]
        if contract.get("width") and width != contract["width"]:
            raise ValueError(f"Unexpected image width: {width}")
        if contract.get("height") and height != contract["height"]:
            raise ValueError(f"Unexpected image height: {height}")
        # An almost constant image is not accepted as a completed edit.
        if float(pixels.std()) < 1e-6:
            raise ValueError("A constant blank image is not a completed model output")
        return {"kind": kind, "width": width, "height": height}
    if kind == "video":
        count, variations, metadata = 0, [], None
        for frame, metadata in video_frames(path):
            count += 1
            variations.append(float(frame.std()))
        if not count or not metadata or not metadata.get("fps"):
            raise ValueError("The video has no decodable frames or frame rate")
        if count != contract["frames"]:
            raise ValueError(f"Expected {contract['frames']} frames, decoded {count}")
        if abs(metadata["fps"] - contract["fps"]) > 0.05:
            raise ValueError(f"Unexpected output fps: {metadata['fps']}")
        if max(variations) < 1e-6:
            raise ValueError("A blank video is not a valid output")
        width, height = metadata["size"]
        for key, actual in [("width", width), ("height", height)]:
            if contract.get(key) and contract[key] != actual:
                raise ValueError(f"Unexpected output {key}: {actual}")
        return {
            "kind": kind,
            "frames": count,
            "fps": metadata["fps"],
            "width": width,
            "height": height,
            "duration_seconds": count / metadata["fps"],
        }
    if kind == "asset":
        import trimesh

        scene = trimesh.load(path, force="scene")
        meshes = list(scene.geometry.values())
        face_count = 0
        for mesh in meshes:
            face_count += len(mesh.faces)
        if not meshes or not face_count:
            raise ValueError("The exported asset has no usable faces")
        for mesh in meshes:
            if not np.isfinite(mesh.vertices).all():
                raise ValueError("The asset has nonfinite vertices")
        # Require texture data, rather than counting a shape-only GLB as a textured result.
        textured = []
        for mesh in meshes:
            if mesh.visual.kind == "texture":
                textured.append(mesh)
        if not textured:
            raise ValueError("The required texture/material stage is missing")
        has_texture = False
        for mesh in textured:
            material = mesh.visual.material
            if getattr(material, "baseColorTexture", None) is not None:
                has_texture = True
                break
            if getattr(material, "image", None) is not None:
                has_texture = True
                break
        if not has_texture:
            raise ValueError("The asset has no reopened base-colour texture")
        if contract.get("turntable_required"):
            validate_output(
                path.parent / "preview.png",
                {"kind": "image", "width": 640, "height": 640},
            )
            validate_output(
                path.parent / "turntable.mp4",
                {"kind": "video", "frames": 36, "fps": 12, "width": 640, "height": 640},
            )
            preview = json.loads((path.parent / "preview.json").read_text())
            if (
                preview.get("frames") != 36
                or preview.get("fps") != 12
                or preview.get("resolution") != [640, 640]
            ):
                raise ValueError("The standardized asset preview contract differs")
        return {
            "kind": kind,
            "faces": face_count,
            "bytes": path.stat().st_size,
            "textured": True,
        }
    raise ValueError(f"Unknown output kind: {kind}")
