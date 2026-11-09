# Read existing GLBs and render the same extra low-angle views for both models.
# Write separately labelled PNGs and camera/source metadata.
# These are inspection renders of saved assets, not new model-generated samples.

"""Render extra low-angle inspection views from existing GLBs only.

Use after measured model runs. This does not generate another model sample or
replace the standard preview, video or GLB. The same four views apply to both models.
"""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import argparse
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.io import atomic_json, sha256, utc_now


def camera_pose(angle):
    """Use the standard preview camera orientation, with the eye below centre."""
    eye = np.array([2 * math.sin(angle), -0.8, 2 * math.cos(angle)])
    forward = eye / np.linalg.norm(eye)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, forward])
    pose[:3, 3] = eye
    return pose


def render_inspection(asset_path, renderer):
    """Read one saved asset and write four separately labelled inspection PNGs."""
    import pyrender
    import trimesh
    from PIL import Image
    from a1.media import validate_output

    asset_path = Path(asset_path).resolve()
    output = asset_path.parent / "inspection"
    output.mkdir()  # Never overwrite an existing inspection.
    source_hash = sha256(asset_path)
    standard_files = [asset_path.parent / name for name in
                      ["preview.png", "preview.json", "turntable.mp4"]]
    standard_hashes = {str(path): sha256(path) for path in standard_files if path.is_file()}
    record = {
        "started_at_utc": utc_now(), "status": "started",
        "source_glb": str(asset_path), "source_glb_sha256": source_hash,
        "scope": "Derived views of a saved asset; no model inference or new model sample",
        "normalization": "bounding box centred; maximum extent scaled to one; saved GLB unchanged",
        "resolution": [640, 640],
        "camera": {"projection": "orthographic", "xmag": 0.8, "ymag": 0.8,
                   "znear": 0.01, "zfar": 10, "eye_height": -0.8, "radius": 2,
                   "look_at": [0, 0, 0], "yaw_degrees": [0, 90, 180, 270]},
        "lighting": "white ambient 0.45 plus camera-aligned directional light 2.5",
        "background": [1.0, 1.0, 1.0, 1.0],
        "views": [], "standard_metrics_changed": False,
        "limit": "Additional visual inspection only; no reference geometry or physical accuracy claim",
    }
    atomic_json(output / "metadata.json", record)
    try:
        asset = trimesh.load(asset_path, force="scene")
        centre = asset.bounds.mean(axis=0)
        extent = float((asset.bounds[1] - asset.bounds[0]).max())
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError("Cannot normalize an empty or degenerate asset")
        normalization = np.eye(4)
        normalization[:3, :3] /= extent
        normalization[:3, 3] = -centre / extent
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                                ambient_light=[0.45, 0.45, 0.45])
        for node in asset.graph.nodes_geometry:
            transform, geometry = asset.graph[node]
            scene.add(pyrender.Mesh.from_trimesh(asset.geometry[geometry], smooth=False),
                      pose=normalization @ transform)
        camera = scene.add(pyrender.OrthographicCamera(
            xmag=0.8, ymag=0.8, znear=0.01, zfar=10))
        light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5))
        for degrees in [0, 90, 180, 270]:
            pose = camera_pose(math.radians(degrees))
            scene.set_pose(camera, pose)
            scene.set_pose(light, pose)
            pixels, _ = renderer.render(scene)
            image = output / f"low_yaw_{degrees:03d}.png"
            Image.fromarray(pixels).save(image)
            validation = validate_output(image, {"kind": "image", "width": 640, "height": 640})
            record["views"].append({"file": image.name, "yaw_degrees": degrees,
                                    "eye": pose[:3, 3].tolist(), "sha256": sha256(image),
                                    "validation": validation})
        if len(record["views"]) != 4:
            raise ValueError("Four valid inspection views are required")
        if sha256(asset_path) != source_hash or any(
                sha256(Path(path)) != digest for path, digest in standard_hashes.items()):
            raise ValueError("A source asset or standard preview changed during inspection")
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at_utc"] = utc_now()
        atomic_json(output / "metadata.json", record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", type=Path,
                        help="Existing GLBs from either model; no model is loaded")
    args = parser.parse_args()
    assets = [path.expanduser().resolve() for path in args.assets]
    outputs = [path.parent / "inspection" for path in assets]
    if len(set(outputs)) != len(outputs):
        parser.error("Use one GLB per attempt folder; repeated inspection destinations are refused")
    for path, output in zip(assets, outputs):
        if path.suffix.lower() != ".glb" or not path.is_file():
            parser.error(f"Missing saved GLB: {path}")
        if output.exists() or output.is_symlink():
            parser.error(f"Inspection already exists: {output}")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not hasattr(np, "infty"):
        np.infty = np.inf  # Compatibility with the same pyrender used for standard previews.
    import pyrender

    renderer = pyrender.OffscreenRenderer(640, 640)
    try:
        for asset in assets:
            print("Saved derived inspection:", render_inspection(asset, renderer), flush=True)
    finally:
        renderer.delete()


if __name__ == "__main__":
    main()
