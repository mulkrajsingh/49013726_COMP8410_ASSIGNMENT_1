# Read the native TRELLIS mesh and export settings; save a recoverable checkpoint.
# Keep the mesh before simplification and the official textured-export step.
# The checkpoint preserves a generated sample; it does not generate another one.

"""Keep the returned TRELLIS mesh before simplification and textured export."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .io import atomic_json, read_json, sha256, utc_now


def _json_value(value):
    """Convert small metadata values without saving Python objects in JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        return _json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError("Unsupported mesh checkpoint metadata: " + type(value).__name__)


def save_mesh_checkpoint(mesh, folder, sampler_metadata, export_settings=None):
    """Save CPU tensors and JSON before calling mesh.simplify or to_glb."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Keep the existing mesh checkpoint: " + str(folder))
    settings = dict(export_settings or {})
    for name in ("vertices", "faces", "attr_volume", "coords", "attr_layout", "voxel_size"):
        if name in settings:
            raise ValueError("export_settings must not replace mesh field " + name)
    settings.setdefault("aabb", [[-0.5] * 3, [0.5] * 3])
    record = {
        "format_version": 1,
        "created_at": utc_now(),
        "stage": "native_mesh_before_adapter_simplification_and_export",
        "tensor_file": "mesh.pt",
        "layout": {
            name: [channels.start, channels.stop, channels.step]
            for name, channels in mesh.layout.items()
        },
        "voxel_size": _json_value(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "sampler_metadata": _json_value(sampler_metadata),
        "export_settings": _json_value(settings),
    }
    # Reject invalid metadata before copying a large mesh off the GPU.
    json.dumps(record, allow_nan=False)
    import torch

    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mesh_checkpoint-", dir=folder.parent))
    try:
        tensors = {}
        for name in ("vertices", "faces", "attrs", "coords", "origin"):
            # Clone also avoids serializing an oversized backing storage/view.
            tensors[name] = getattr(mesh, name).detach().cpu().contiguous().clone()
        record["origin"] = tensors["origin"].tolist()
        record["tensors"] = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": tensor.numel() * tensor.element_size(),
            }
            for name, tensor in tensors.items()
        }
        record["total_tensor_bytes"] = sum(item["bytes"] for item in record["tensors"].values())
        tensor_file = temporary / "mesh.pt"
        with tensor_file.open("wb") as stream:
            torch.save(tensors, stream)
            stream.flush()
            os.fsync(stream.fileno())
        record["tensor_file_bytes"] = tensor_file.stat().st_size
        record["tensor_sha256"] = sha256(tensor_file)
        atomic_json(temporary / "mesh.json", record)
        # Only publish the directory once both complete files are present.
        os.replace(temporary, folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def load_mesh_checkpoint(folder, device="cpu"):
    """Verify and load native constructor fields without loading any model."""
    folder = Path(folder)
    record = read_json(folder / "mesh.json")
    if record["format_version"] != 1:
        raise ValueError("Unsupported mesh checkpoint version")
    tensor_file = folder / "mesh.pt"
    if sha256(tensor_file) != record["tensor_sha256"]:
        raise ValueError("Mesh checkpoint tensor checksum differs")
    import torch

    tensors = torch.load(tensor_file, map_location="cpu", weights_only=True)
    fields = {
        name: tensors[name].to(device)
        for name in ("vertices", "faces", "attrs", "coords")
    }
    fields.update({
        "origin": record["origin"],
        "voxel_size": record["voxel_size"],
        "voxel_shape": torch.Size(record["voxel_shape"]),
        "layout": {name: slice(*channels) for name, channels in record["layout"].items()},
    })
    return fields, record


def restore_export_kwargs(folder, device="cpu"):
    """Restore to_glb inputs; pass device='cuda' for the native GPU exporter."""
    fields, record = load_mesh_checkpoint(folder, device=device)
    arguments = dict(record["export_settings"])
    arguments.update({
        "vertices": fields["vertices"],
        "faces": fields["faces"],
        "attr_volume": fields["attrs"],
        "coords": fields["coords"],
        "attr_layout": fields["layout"],
        "voxel_size": fields["voxel_size"],
    })
    return arguments
