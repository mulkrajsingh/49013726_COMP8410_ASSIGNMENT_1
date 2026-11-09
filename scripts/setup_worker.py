# Read one worker's model registry entry; print or execute its setup recipe.
# Explicit setup can install third-party dependencies and record exact revisions.
# Setup success does not establish that a model run or evaluation succeeded.

"""Print a worker recipe, or execute it inside an explicitly selected environment.

This script never creates a cloud instance, accepts model terms, or starts inference.
Read the generated recipe and upstream instructions before using --execute.
"""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.io import atomic_json, read_json, utc_now


# Build the installation commands for the selected model implementation.
# Different model pipelines need separate dependency sets.
def recipe(model):
    """Return installation commands and notes for the supplied model settings.

    Select dependencies by implementation and include registered components.
    This only builds a dictionary; it does not install packages, resolve
    versions, create a machine or run any model."""
    # Install into the selected worker interpreter rather than a shared environment.
    python = sys.executable
    pip = [python, "-m", "pip", "install"]
    kind = model["implementation"]
    common = pip + [
        "numpy",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "huggingface_hub",
        "ipykernel",
        "trimesh",
        "psutil>=5.9,<8",
        "nvidia-ml-py>=12,<14",
    ]
    if kind in {"trellis", "hunyuan3d"}:
        common += ["pyrender"]
    if kind in {"qwen_vlm", "gemma_vlm", "qwen_agent", "smollm_agent", "qwen_image", "flux_image"}:
        # Reuse the successful general worker versions from the experiment record.
        common = pip + ["numpy==2.1.2", "pillow==11.0.0", "imageio==2.38.1",
                        "imageio-ffmpeg==0.7.0", "psutil==7.1.0", "nvidia-ml-py==13.615.71"]
        commands = [
            pip + ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
                   "--index-url", "https://download.pytorch.org/whl/cu128"],
            pip + ["transformers==5.20.2", "tokenizers==0.23.1", "accelerate==1.15.0",
                   "huggingface-hub==1.33.0", "safetensors==0.8.0", "sentencepiece==0.2.1",
                   "git+https://github.com/huggingface/diffusers.git@51349f7ac2da4853852efdc1befc92669946e68a"],
            common,
        ]
    elif kind == "trellis":
        # Match the tested Python/Torch pair and keep the original checkout clean.
        pip = pip + ["--no-compile"]
        native_install = """import json, os, shutil, subprocess, sys
from pathlib import Path
root = Path(%r)
persistent = Path(os.environ.get('A1_PERSISTENT_ROOT', str(root.parent)))
sources = persistent / 'third_party_extensions/P03_A'
sources.mkdir(parents=True, exist_ok=True)
extensions = [
    ('nvdiffrast', 'https://github.com/NVlabs/nvdiffrast.git', '253ac4fcea7de5f396371124af597e6cc957bfae'),
    ('nvdiffrec', 'https://github.com/JeffreyXiang/nvdiffrec.git', 'b296927cc7fd01c2ac1087c8065c4d7248f72da4'),
    ('CuMesh', 'https://github.com/JeffreyXiang/CuMesh.git', '12289e1062f0603f2f0d0771b02e1395d247f26f'),
    ('FlexGEMM', 'https://github.com/JeffreyXiang/FlexGEMM.git', '6dd94a859c26ee8246888502eada3dd8ad85532e'),
]
revisions = {}
for name, url, revision in extensions:
    folder = sources / name
    if not folder.exists():
        subprocess.run(['git', 'clone', url, str(folder)], check=True)
    subprocess.run(['git', 'checkout', '--detach', revision], cwd=folder, check=True)
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=folder, check=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=folder, text=True).strip()
    if actual != revision:
        raise ValueError('Native extension revision differs: ' + name)
    revisions[name] = actual
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-compile', '--no-build-isolation', str(folder)], check=True)
record = root / 'evidence/setup/P03_A_extension_revisions.json'
record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps(revisions, indent=2) + '\\n')
voxel = sources / 'o-voxel'
if not voxel.exists():
    shutil.copytree(Path.cwd() / 'o-voxel', voxel)
# These dependencies were installed above. Avoid O-Voxel's unpinned Git URLs.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-compile', '--no-build-isolation', '--no-deps', str(voxel)], check=True)
""" % str(ROOT)
        commands = [
            pip + ["setuptools<81", "wheel", "packaging", "ninja"],
            pip + [
                "torch==2.6.0", "torchvision==0.21.0",
                "--index-url", "https://download.pytorch.org/whl/cu124",
            ],
            pip + [
                "imageio", "imageio-ffmpeg", "tqdm", "easydict",
                "opencv-python-headless", "trimesh", "transformers==4.57.6",
                "gradio==6.0.1", "tensorboard", "pandas", "lpips",
                "zstandard", "kornia", "timm",
            ],
            pip + [
                "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
            ],
            pip + ["pillow-simd==12.1.1.post0"],
            pip + ["flash-attn==2.7.3", "--no-build-isolation"],
            [python, "-c", native_install],
            common,
        ]
    elif kind == "hunyuan3d":
        # Keep upstream pins, but fetch Blender's original wheel from its archive.
        runtime_requirements = ROOT / "evidence/setup/P03_B_runtime_requirements.txt"
        prepare_requirements = """from pathlib import Path
source = Path('requirements.txt')
target = Path(%r)
target.parent.mkdir(parents=True, exist_ok=True)
lines = []
for line in source.read_text().splitlines():
    if line.strip().startswith('--extra-index-url'):
        continue
    if line.strip() == 'bpy==4.0':
        line = 'bpy @ https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl'
    lines.append(line)
target.write_text('\\n'.join(lines) + '\\n')
""" % str(runtime_requirements)
        commands = [
            pip + ["setuptools<81", "wheel", "packaging"],
            pip
            + [
                "torch==2.5.1",
                "torchvision==0.20.1",
                "torchaudio==2.5.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ],
            [python, "-c", prepare_requirements],
            pip
            + [
                "--index-url", "https://pypi.org/simple",
                "--no-build-isolation", "-r", str(runtime_requirements),
            ],
            pip + ["--no-build-isolation", "-e", "hy3dpaint/custom_rasterizer"],
            [python, str(ROOT / "scripts/build_mesh_painter.py")],
        ]
    else:
        raise ValueError(kind)
    model_ids = [model["model_id"]]
    for component in model["components"]:
        model_ids.append(component["model_id"])
    return {
        "worker": model["worker"],
        "tested": False,
        "requires": "Linux NVIDIA CUDA; native build toolkit and EGL for asset workers",
        "environment_note": "P01/P02/P04 used the recorded shared Python 3.12 general environment. P03_A and P03_B used separate Python 3.10 native environments. See evidence/environments and the GPU reproduction guide. Installation is printed for review; these revised setup commands have not been rerun on a GPU.",
        "recorded_environment": "evidence/environments/" + model["worker"] + ".json",
        "source_repository": model["code_repository"],
        "python_requirement": "3.10" if kind in {"trellis", "hunyuan3d"} else "3.12",
        "install_commands": commands,
        "common_command": common,
        "models": model_ids,
    }


# Resolve the exact source code and weight versions used by a worker.
# Later runs need the same original model and component versions.
def resolve_revisions(worker):
    """Resolve worker source revisions and return its updated model record.

    Use the worker name to read the registry, clone/check out upstream code
    when required, and query exact model/component revisions. Update
    config/models.json and evidence/setup/<worker>.json. This contacts source
    services and prepares code; successful inference still needs a pilot."""
    from huggingface_hub import model_info

    registry = read_json(ROOT / "config/models.json")
    model = registry[worker]
    repo = ROOT / "third_party" / worker
    if model["code_repository"]:
        if not repo.exists():
            repo.parent.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--recursive",
                    "https://github.com/" + model["code_repository"] + ".git",
                    str(repo),
                ],
                check=True,
            )
        if model["code_revision"]:
            subprocess.run(
                ["git", "checkout", "--detach", model["code_revision"]],
                cwd=repo,
                check=True,
            )
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=repo,
            check=True,
        )
        model["code_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    model["weight_revision"] = model_info(
        model["model_id"], revision=model["weight_revision"] or "main"
    ).sha
    for component in model["components"]:
        component["revision"] = model_info(
            component["model_id"], revision=component["revision"] or "main"
        ).sha
    # Resolution proves a source version exists; it does not certify weight access or SOTA.
    atomic_json(ROOT / "config/models.json", registry)
    atomic_json(
        ROOT / f"evidence/setup/{worker}.json",
        {
            "resolved_at": utc_now(),
            "model": model,
            "status": "sources_resolved_not_piloted",
            "licence_terms_accepted_by_script": False,
        },
    )
    return model


# Print the setup plan unless execution was explicitly requested.
# The same script can be inspected locally before installing on the intended GPU worker.
def main():
    """Read CLI arguments and inspect or execute the chosen worker setup.

    Print the recipe by default. Only --execute on Linux runs installation
    and revision resolution; --resolve-only stops after resolving versions.
    Return None. Execution writes setup records and can install dependencies,
    but never starts inference or creates a cloud instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resolve-only", action="store_true")
    arguments = parser.parse_args()
    model = read_json(ROOT / "config/models.json")[arguments.worker]
    instructions = recipe(model)
    # Print the plan before any installation or revision lookup happens.
    if not arguments.execute:
        print(json.dumps(instructions, indent=2))
        return
    if platform.system() != "Linux":
        raise SystemExit(
            "Execute worker installation on the intended isolated Linux GPU machine; use inspection locally"
        )
    install_env = os.environ.copy()
    if model["implementation"] == "hunyuan3d":
        if sys.version_info[:2] != (3, 10):
            raise SystemExit("Use the isolated Hunyuan Python 3.10 environment")
        # Native build scripts invoke python by name, so use this same worker.
        install_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + install_env.get("PATH", "")
        install_env["VIRTUAL_ENV"] = sys.prefix
        install_env["PIP_INDEX_URL"] = "https://pypi.org/simple"
        install_env["PIP_EXTRA_INDEX_URL"] = ""
    elif model["implementation"] == "trellis":
        if sys.version_info[:2] != (3, 10):
            raise SystemExit("Use the isolated TRELLIS Python 3.10 environment")
        install_env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + install_env.get("PATH", "")
        install_env["VIRTUAL_ENV"] = sys.prefix
        install_env["PIP_INDEX_URL"] = "https://pypi.org/simple"
        install_env["PIP_EXTRA_INDEX_URL"] = ""
        install_env["PIP_NO_COMPILE"] = "1"
        install_env["PIP_COMPILE"] = "0"
        install_env["PIP_NO_DEPS"] = "0"
        install_env["MAX_JOBS"] = install_env.get("MAX_JOBS", "4")
        if install_env.get("CUDA_HOME"):
            install_env["PATH"] = str(Path(install_env["CUDA_HOME"]) / "bin") + os.pathsep + install_env["PATH"]
        if not install_env.get("TORCH_CUDA_ARCH_LIST"):
            # L40 reports 8.9; query the selected machine instead of hard-coding it.
            capabilities = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
            ).splitlines()
            install_env["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(set(capabilities)))
        persistent = Path(install_env.get("A1_PERSISTENT_ROOT", str(ROOT.parent)))
        for name, folder in [
            ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
            ("TRITON_CACHE_DIR", "triton"),
            ("CUDA_CACHE_PATH", "cuda"),
        ]:
            location = persistent / "cache" / folder / "P03_A"
            location.mkdir(parents=True, exist_ok=True)
            install_env[name] = str(location)
        # FlexGEMM uses this fixed home path for its own autotune results.
        import shutil

        flex_cache = persistent / "cache/flex_gemm/P03_A"
        flex_cache.mkdir(parents=True, exist_ok=True)
        default_cache = Path.home() / ".flex_gemm"
        if default_cache.exists() and not default_cache.is_symlink():
            shutil.copytree(default_cache, flex_cache, dirs_exist_ok=True)
            default_cache.rename(
                default_cache.with_name(".flex_gemm_before_COMP8410_" + str(os.getpid()))
            )
        if not default_cache.exists():
            default_cache.symlink_to(flex_cache, target_is_directory=True)
        if default_cache.resolve() != flex_cache.resolve():
            raise RuntimeError("Review the existing FlexGEMM cache symlink before continuing")
    bootstrap = [sys.executable, "-m", "pip", "install", "huggingface_hub"]
    subprocess.run(bootstrap, check=True, env=install_env)
    model = resolve_revisions(arguments.worker)
    if arguments.resolve_only:
        return
    cwd = ROOT
    if model["code_repository"]:
        cwd = ROOT / "third_party" / arguments.worker
    for command in instructions["install_commands"]:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=cwd, check=True, env=install_env)
    if model["implementation"] == "hunyuan3d":
        from a1.data import download

        record = download(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            cwd / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
            100 * 1024 * 1024,
        )
        atomic_json(ROOT / f"evidence/setup/{arguments.worker}_auxiliary.json", record)
    print(
        "Activate the resulting worker environment, install the common packages printed by the inspection recipe, and run its notebook."
    )


if __name__ == "__main__":
    main()
