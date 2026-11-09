# Adapt each recorded model API to the worker's inputs and output files.
# Read the pinned model settings, then save raw replies, images or 3D assets.
# The released libraries implement the pretrained models; these are evaluation adapters.

"""Adapters to the pinned model APIs used by the saved GPU experiments.

See references/implementation_sources.json and NOTICE.md for the upstream APIs.
The adapters implement evaluation I/O; they do not reimplement model internals.
"""

import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

from .io import atomic_json, project_path


# Temporarily run inside the original model code folder.
# Some official loaders expect their own files to be available through relative paths.
@contextlib.contextmanager
def repository_directory(path):
    """Temporarily use path as the original model code working directory."""
    previous = Path.cwd()
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(path))


# Keep the selected model loaded and adapt its input and output formats.
# The shared worker can use different models without changing the experiment procedure.
class ModelRunner:
    """Keep Python pipelines loaded between jobs in a single worker kernel."""

    def __init__(self, root, model):
        """Keep root and model settings for later loading and generation."""
        self.root = Path(root)
        self.model = model
        self.parameters = model["parameters"]
        self.kind = model["implementation"]
        self.repo = self.root / "third_party" / model["worker"]
        self.timing_boundary = "pipeline call after persistent model load; includes saved output and 3D export/preview"

    # Locate or download the exact recorded model revision.
    # A later change to the online weights should not silently change the experiment.
    def local_weights(self, model_id=None):
        """Return the local path of a model snapshot at its recorded revision."""
        from huggingface_hub import snapshot_download

        model_id = model_id or self.model["model_id"]
        if model_id == self.model["model_id"]:
            revision = self.model["weight_revision"]
            patterns = self.model.get("download_patterns")
        else:
            component = None
            for candidate in self.model["components"]:
                if candidate["model_id"] == model_id:
                    component = candidate
                    break
            if component is None:
                raise StopIteration
            revision = component["revision"]
            patterns = component.get("download_patterns")
        if not revision or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError(
                f"Resolve and record the immutable revision for {model_id} before its pilot"
            )
        return Path(
            snapshot_download(model_id, revision=revision, allow_patterns=patterns)
        )

    # Check the original code commit and look for local changes.
    # The run must use the code version recorded in its model configuration.
    def verify_repository(self):
        """Check this runner upstream Git checkout against its saved revision."""
        if not self.repo.is_dir():
            raise FileNotFoundError(
                f"Install this worker's official code first: {self.repo}"
            )
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        if actual != self.model["code_revision"]:
            raise ValueError(
                "The native code checkout differs from the recorded revision"
            )
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=self.repo, text=True
        ).strip():
            raise ValueError(
                "The official checkout has local changes; record or restore them before a pilot"
            )

    # Force model loaders to use the recorded component revisions.
    # Unlisted downloads could otherwise change part of a pipeline without being noticed.
    def _pin_native_hub_downloads(self):
        """Make the upstream loaders use this model recorded component revisions."""
        import huggingface_hub

        revisions = {self.model["model_id"]: self.model["weight_revision"]}
        for component in self.model["components"]:
            revisions[component["model_id"]] = component["revision"]
        for name in ["hf_hub_download", "snapshot_download"]:
            original = getattr(huggingface_hub, name)

            # Check each model download request against the saved revision list.
            # This stops an official loader from silently fetching a different component version.
            def pinned(*args, _original=original, **kwargs):
                """Check a native download request and call its original library function."""
                default_repo_id = None
                if args:
                    default_repo_id = args[0]
                repo_id = kwargs.get("repo_id", default_repo_id)
                if kwargs.get("repo_type", "model") == "model":
                    revision = revisions.get(repo_id)
                    if not revision:
                        raise ValueError(
                            f"Unpinned model component {repo_id}; register and resolve it before continuing"
                        )
                    if kwargs.get("revision") not in (None, "main", revision):
                        raise ValueError(
                            f"A native loader requested a different revision of {repo_id}"
                        )
                    kwargs["revision"] = revision
                return _original(*args, **kwargs)

            setattr(huggingface_hub, name, pinned)

    # Load the original pretrained pipeline for the selected implementation.
    # The persistent worker reuses the model across its planned inputs.
    def load(self):
        """Load the selected pretrained model into this runner for repeated jobs."""
        import torch

        if self.model["code_repository"]:
            self.verify_repository()
        # TRELLIS keeps the mixed precision defined by its own checkpoint configs.
        dtype_name = self.parameters.get("dtype")
        dtype = getattr(torch, dtype_name) if dtype_name else None
        if self.kind in {"qwen_agent", "smollm_agent"}:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            weights = str(self.local_weights())
            self.tokenizer = AutoTokenizer.from_pretrained(weights)
            self.pipeline = AutoModelForCausalLM.from_pretrained(
                weights, dtype=dtype, device_map="auto",
                disable_mmap=self.parameters.get("disable_mmap", False),
            )
        elif self.kind in {"qwen_vlm", "gemma_vlm"}:
            from transformers import AutoProcessor

            if self.kind == "qwen_vlm":
                from transformers import Qwen3_5ForConditionalGeneration as Model
            else:
                from transformers import AutoModelForMultimodalLM as Model
            weights = str(self.local_weights())
            self.processor = AutoProcessor.from_pretrained(weights)
            self.pipeline = Model.from_pretrained(
                weights, dtype=dtype, device_map="auto",
                disable_mmap=self.parameters.get("disable_mmap", False),
            )
        elif self.kind in {"qwen_image", "flux_image"}:
            if self.kind == "qwen_image":
                from diffusers import QwenImage21Pipeline as Pipeline
            else:
                from diffusers import Flux2KleinPipeline as Pipeline
            self.pipeline = Pipeline.from_pretrained(
                str(self.local_weights()), torch_dtype=dtype
            )
            placement = self.parameters.get("placement", "model_cpu_offload")
            if placement == "cuda":
                self.pipeline.to("cuda")
            elif placement == "model_cpu_offload":
                self.pipeline.enable_model_cpu_offload()
            else:
                raise ValueError("Unknown image pipeline placement")
        elif self.kind == "trellis":
            self._pin_native_hub_downloads()
            sys.path.insert(0, str(self.repo))
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                str(self.local_weights())
            )
            self.pipeline.cuda()
        elif self.kind == "hunyuan3d":
            # BasicSR uses the old module name for this unchanged public function.
            import torchvision.transforms.functional as image_functions
            sys.modules.setdefault("torchvision.transforms.functional_tensor", image_functions)
            # Diffusers copies Python modules into its cache immediately before importing.
            import importlib
            import diffusers.utils.dynamic_modules_utils as dynamic_modules
            original_import = dynamic_modules.get_class_in_module

            def import_new_module(*args, **kwargs):
                importlib.invalidate_caches()
                return original_import(*args, **kwargs)

            dynamic_modules.get_class_in_module = import_new_module
            self._pin_native_hub_downloads()
            sys.path[:0] = [
                str(self.repo),
                str(self.repo / "hy3dshape"),
                str(self.repo / "hy3dpaint"),
            ]
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

            with repository_directory(self.repo):
                self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    str(self.local_weights()), dtype=dtype
                )
                paint_config = Hunyuan3DPaintConfig(
                    max_num_view=self.parameters["paint_views"],
                    resolution=self.parameters["paint_resolution"],
                )
                # The upstream default is relative to a different working folder.
                checkpoint = self.repo.resolve() / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Run this worker's setup first: {checkpoint}")
                paint_config.realesrgan_ckpt_path = str(checkpoint)
                self.paint = Hunyuan3DPaintPipeline(paint_config)
        else:
            raise ValueError(f"Unknown implementation: {self.kind}")

    # Run the selected model on one input and save its required output files.
    # The worker can then validate and record the same output contract for both models.
    def generate(self, job, folder):
        """Run job with the loaded pipeline and save its output inside folder."""
        import torch

        parameters = self.parameters
        item = job["item"]
        folder = Path(folder).resolve()
        seed = job["seed"]
        # Use the recorded seed before any generation call.
        torch.manual_seed(seed)
        image_path = None
        if item.get("image"):
            image_path = project_path(self.root, item["image"])
        # Image-to-3D models use the photograph without a text instruction.
        prompt = item.get("prompt")
        if self.kind in {"qwen_agent", "smollm_agent"}:
            from .agents import run_agent
            from .io import read_json

            catalogue = read_json(project_path(self.root, item["catalogue"]))
            call_number = 0

            # Generate one visible agent reply and count its tokens.
            # The shared tool loop needs the raw reply and the same output limit for both models.
            def complete(messages, tools):
                """Generate one agent reply from messages and the shared tools."""
                nonlocal call_number
                call_number += 1
                original_messages = messages
                if self.kind == "smollm_agent":
                    from .agents import smol_tool_history

                    messages = smol_tool_history(messages)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(
                    self.pipeline.device
                )
                # Evaluation uses pretrained weights without gradient updates.
                with torch.inference_mode():
                    tokens = self.pipeline.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=parameters["max_new_tokens"],
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                # Count only new reply tokens, excluding the input prompt.
                generated = tokens[0][inputs["input_ids"].shape[-1] :]
                # Save every reply before parsing so a later failed turn loses no earlier replies.
                atomic_json(folder / f"turn_{call_number:02d}_raw.json", {
                    "messages": original_messages,
                    "native_prompt": text,
                    "input_token_ids": inputs["input_ids"][0].tolist(),
                    "generated_token_ids": generated.tolist(),
                    "raw_decoded": self.tokenizer.decode(
                        generated, skip_special_tokens=False
                    ),
                })
                return {
                    "text": self.tokenizer.decode(
                        generated, skip_special_tokens=True
                    ).strip(),
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "output_tokens": len(generated),
                    "possibly_truncated": len(generated)
                    >= parameters["max_new_tokens"],
                }

            trace = run_agent(
                prompt,
                catalogue,
                complete,
                parameters["max_turns"],
                parameters["max_tool_calls"],
            )
            atomic_json(folder / "agent.json", trace)
            return "agent.json"
        if self.kind in {"qwen_vlm", "gemma_vlm"}:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.open(image_path).convert("RGB"),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.pipeline.device)
            with torch.inference_mode():
                tokens = self.pipeline.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=parameters["max_new_tokens"],
                )
            # Count only new reply tokens, excluding the input prompt.
            generated = tokens[0][inputs["input_ids"].shape[-1] :]
            raw_full = self.processor.decode(generated, skip_special_tokens=False).strip()
            # Keep the original tokens so formatting can be inspected without inference.
            atomic_json(folder / "raw_generation.json", {
                "text": raw_full,
                "input_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "output_token_ids": generated.detach().cpu().tolist(),
            })
            native_response = None
            if self.kind == "gemma_vlm":
                native_response = self.processor.parse_response(
                    raw_full, prefix=inputs["input_ids"][0]
                )
                atomic_json(folder / "native_response.json", native_response)
                raw = native_response.get("content", "").strip()
            else:
                raw = self.processor.decode(generated, skip_special_tokens=True).strip()
            # Preserve this historical format check; the later exact-schema validator is separate.
            try:
                parsed = json.loads(raw)
                answer, parsed_ok = parsed["answer"], isinstance(parsed["answer"], str)
            except (ValueError, KeyError, TypeError):
                answer, parsed_ok = raw, False
            atomic_json(
                folder / "answer.json",
                {
                    "answer": answer,
                    "raw_text": raw,
                    "raw_decoded": raw_full,
                    "native_response": native_response,
                    "structured_parse_success": parsed_ok,
                    "generated_tokens": len(generated),
                    "possibly_truncated": len(generated)
                    >= parameters["max_new_tokens"],
                },
            )
            return "answer.json"
        # The edit mask is used later for scoring, not passed into these pipelines.
        if self.kind in {"qwen_image", "flux_image"}:
            arguments = {
                "prompt": prompt,
                "image": Image.open(image_path).convert("RGB"),
                "width": parameters["width"],
                "height": parameters["height"],
                "num_inference_steps": parameters["steps"],
                "generator": torch.Generator("cuda").manual_seed(seed),
            }
            if self.kind == "flux_image":
                arguments["guidance_scale"] = parameters["guidance"]
            self.pipeline(**arguments).images[0].save(folder / "image.png")
            return "image.png"
        if self.kind == "trellis":
            import o_voxel

            mesh = self.pipeline.run(
                Image.open(image_path).convert("RGB"),
                seed=seed,
                pipeline_type=parameters["pipeline_type"],
            )[0]
            export_settings = {
                "aabb": [[-0.5] * 3, [0.5] * 3],
                "decimation_target": 1000000, "texture_size": 4096,
                "remesh": True, "remesh_band": 1, "remesh_project": 0,
                "verbose": True,
            }
            # Keep the sampled mesh if a later export or preview fails.
            from .mesh_checkpoint import save_mesh_checkpoint

            save_mesh_checkpoint(
                mesh, folder / "raw_mesh",
                {"job_id": job["job_id"], "seed": seed,
                 "pipeline_type": parameters["pipeline_type"],
                 "adapter_simplify_target": 16777216},
                export_settings,
            )
            # Prepare the generated mesh for the official textured export step.
            mesh.simplify(16777216)
            asset = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                **export_settings,
            )
            asset.export(str(folder / "asset.glb"), extension_webp=True)
            from .assets import render_turntable

            # Reopen the exported asset for the shared preview.
            render_turntable(folder / "asset.glb")
            return "asset.glb"
        if self.kind == "hunyuan3d":
            with repository_directory(self.repo):
                mesh = self.pipeline(
                    image=str(image_path),
                    generator=torch.Generator("cuda").manual_seed(seed),
                    num_inference_steps=parameters["steps"],
                )[0]
                mesh.export(folder / "shape.obj")
                self.paint(
                    mesh_path=str(folder / "shape.obj"),
                    image_path=str(image_path),
                    output_mesh_path=str(folder / "asset.obj"),
                    save_glb=True,
                )
            from .assets import render_turntable

            # Reopen the exported asset for the shared preview.
            render_turntable(folder / "asset.glb")
            return "asset.glb"
        raise ValueError(f"Unimplemented model adapter: {self.kind}")
