# GPU reproduction

The recorded environments and model revisions are the reference for reproducing inference. The saved-output analysis in README.md is a separate CPU route. No additional GPU inference was run during submission preparation.

## Original platforms

P03 A/B ran sequentially on NVIDIA L40. P01, P02 and P04 ran on NVIDIA A100 SXM4. P01/P02/P04 shared Python 3.12.3 with Torch 2.8.0 CUDA 12.8, Transformers 5.20.2, Hub 1.33.0 and the recorded Diffusers commit. P03 A used Torch 2.6.0 CUDA 12.4 and Transformers 4.57.6. P03 B used Torch 2.5.1 CUDA 12.4 with the separate native dependencies in its environment record.

Both sides of each pair used the same GPU class. Hardware, driver, package locks and source revisions are in `evidence/environments/`, `evidence/setup/` and `config/models.json`. GPU device memory, Torch allocator memory and summed process RSS describe different quantities.

## Prepare the selected environment

Use Linux x86_64 with a compatible NVIDIA driver/CUDA toolkit. Make a fresh Python 3.12 environment for the six general workers, or a Python 3.10 environment for each P03 worker. Read the exact recorded requirements before installing. The full freezes include system-specific packages and must not be blindly installed on macOS.

Print the installation recipe using the selected environment's Python:

```bash
python scripts/setup_worker.py --worker P01_A
python scripts/setup_worker.py --worker P03_A
python scripts/setup_worker.py --worker P03_B
```

Only run one chosen command at a time. `--execute` installs the recipe on Linux; it does not run inference. Existing model revisions are pinned in the registry. Do not switch them to moving `main` revisions when reproducing the study. The six general-worker recipes now use the recorded runtime versions rather than unbounded latest versions.

For P03, keep the official code checkouts at their recorded commits. Native compilation needs a C/C++ toolchain, the matching Python headers and CUDA toolkit. `scripts/build_mesh_painter.py` handles the Hunyuan painter extension. Its RealESRGAN checkpoint is fetched by setup with its source receipt.

## Native dependency recovery

`evidence/reproduction/P03_A_native_wheels/` retains the eight compiled/Python extension wheels and the exact runtime list from the successful reproduction. They target CPython 3.10, Linux x86_64 and the recorded Torch/CUDA combination. Install the matching Torch 2.6.0 CUDA 12.4 first, then the supplied wheels with `--no-deps`; install the remaining recorded dependencies separately. The old absolute `/workspace/...` paths in the runtime list describe where files originally lived. Use the packaged wheel paths on the new machine.

`evidence/reproduction/P03_B_native_artifacts/` retains the rasterizer wheel and compiled mesh painter from the successful environment. Rebuild from the pinned source when the machine ABI differs. The recorded preparation drivers, import checks, command logs and binary manifests remain beside these files. Pillow-SIMD ordering, the BasicSR torchvision compatibility alias, dynamic module cache invalidation and EGL renderer isolation are documented in `evidence/issues_and_fixes.json`.

P03 uses `pyrender` and an EGL-capable graphics stack for the shared preview. The renderer runs in a separate process. It reopens each GLB and writes the same 36-frame, 640 by 640, 12 fps turntable without changing saved geometry. Shape, texture, export and preview are part of the recorded generation boundary.

## Inputs and execution

Keep the supplied study settings and prepared inputs unchanged. Arrange any required model access yourself; credentials and weights are not included. Token values must not be written into notebooks or logs.

Inspect a worker first:

```bash
python scripts/run_worker.py --worker P01_A --phase pilot
```

Read the script help and notebook before adding `--execute`. A normal persistent project folder can be used. `A1_PERSISTENT_ROOT` optionally selects a cache location; a mounted RunPod volume and Mac collector are not required by the revised runner. A separate `--backup-root` is optional. The original Mac-copy collector protocol remains documented in historical evidence.

A matching saved successful job is reused. To generate a fresh independent experiment, work in a separate copy without the old outputs. New or failed attempts are stored separately, not over the original output. Do not rerun a valid output simply because its appearance or answer is unattractive.

## Reproduction evidence and limits

The supplied fresh-environment repetitions are the original P01 and P03 pilot repetitions. P01 answer and token content matched the originals. P03 repeated outputs reopened as textured GLBs but were not byte-identical. Model weights and some native/runtime caches were reused, so this is not a fully independent hardware or package-download reproduction.

The P03 A historical pilot has an explicitly documented renderer-source ambiguity in `evidence/pilot_source_history/`. It is retained as historical evidence with that limitation. Main runs and later reproductions use the recorded stable source. Reproducing an output does not establish its factual or geometric accuracy.
