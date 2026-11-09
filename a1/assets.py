# Read a generated GLB; save a standard preview and rotating video.
# Both 3D model outputs use the same camera and lighting for inspection.
# A rendered view does not establish the object's true geometry or materials.

"""Render the same saved-GLB turntable for both product generation workflows."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .io import atomic_json


# Reopen the saved 3D asset and render a fixed rotation around it.
# Both models need the same inspection views without modifying the saved geometry.
def render_turntable(asset_path):
    # Use a fresh renderer process after the combined process had cleanup crashes.
    subprocess.run(
        [sys.executable, "-m", "a1.assets", str(Path(asset_path).resolve())],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )
    return "turntable.mp4"


def _render_turntable(asset_path):
    # Set the EGL backend before importing OpenGL on the Linux CUDA worker.
    """Render standard inspection views of asset_path and return the video name."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # pyrender 0.1.45 still uses this old name, removed in NumPy 2.
    if not hasattr(np, "infty"):
        np.infty = np.inf
    import imageio.v2 as imageio
    import pyrender
    import trimesh
    from PIL import Image

    asset_path = Path(asset_path)
    asset = trimesh.load(asset_path, force="scene")
    centre = asset.bounds.mean(axis=0)
    extent = float((asset.bounds[1] - asset.bounds[0]).max())
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Cannot normalize an empty or degenerate asset")
    # Apply this only to the preview scene, not the saved mesh.
    normalization = np.eye(4)
    normalization[:3, :3] /= extent
    normalization[:3, 3] = -centre / extent
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
    for node in asset.graph.nodes_geometry:
        transform, geometry = asset.graph[node]
        scene.add(
            pyrender.Mesh.from_trimesh(asset.geometry[geometry], smooth=False),
            pose=normalization @ transform,
        )
    camera = scene.add(
        pyrender.OrthographicCamera(xmag=0.8, ymag=0.8, znear=0.01, zfar=10)
    )
    light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5))
    renderer = pyrender.OffscreenRenderer(640, 640)
    try:
        with imageio.get_writer(
            asset_path.parent / "turntable.mp4",
            fps=12,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            # Use the same evenly spaced views for both 3D models.
            for frame in range(36):
                angle = 2 * math.pi * frame / 36
                eye = np.array([2 * math.sin(angle), 0.8, 2 * math.cos(angle)])
                forward = eye / np.linalg.norm(eye)
                right = np.cross([0, 1, 0], forward)
                right /= np.linalg.norm(right)
                up = np.cross(forward, right)
                pose = np.eye(4)
                pose[:3, :3] = np.column_stack([right, up, forward])
                pose[:3, 3] = eye
                scene.set_pose(camera, pose)
                scene.set_pose(light, pose)
                pixels, _ = renderer.render(scene)
                # Save one still image as well as the rotating video.
                if frame == 0:
                    Image.fromarray(pixels).save(asset_path.parent / "preview.png")
                writer.append_data(pixels)
    finally:
        renderer.delete()
    atomic_json(
        asset_path.parent / "preview.json",
        {
            "normalization": "bounding box centred; maximum extent scaled to one; saved GLB unchanged",
            "camera": "orthographic; xmag=ymag=0.8; 36 equally spaced yaw views; eye height 0.8 and radius 2",
            "resolution": [640, 640],
            "frames": 36,
            "fps": 12,
            "lighting": "white ambient 0.45 plus camera-aligned directional light 2.5",
            "limit": "standardized inspection only; no reference geometry or geometric accuracy claim",
        },
    )
    return "turntable.mp4"


if __name__ == "__main__":
    _render_turntable(sys.argv[1])
