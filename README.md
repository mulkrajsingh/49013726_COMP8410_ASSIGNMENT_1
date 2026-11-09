# COMP8410 Assignment 1

Student ID: 49013726. Four paired evaluations for a hypothetical IKEA retail workflow.

| Pair | Task | A | B | Final outputs |
|---|---|---|---|---:|
| P01 | Visible product facts | Qwen3.5 4B | Gemma 4 E2B IT | 30 |
| P02 | Product photo background editing | Qwen Image 2.1 | FLUX.2 klein base 4B | 12 |
| P03 | Single image to textured 3D | TRELLIS.2 4B | Hunyuan3D 2.1 | 12 |
| P04 | Retail requests with read-only tools | Qwen3 4B Instruct 2507 | SmolLM3 3B | 24 |

The supplied evidence contains 78 final outputs, eight pilots and four fresh-environment reproductions. Two failed Hunyuan pilot attempts are also retained. A completed run means that its files were saved; quality and strict task acceptance are separate measurements.

I compare the models as separate components of two proposed retail systems. The experiment does not implement an integrated IKEA service. Photographs come from Amazon Berkeley Objects (ABO), and catalogue prices, stock and returns policies are fictional.

## Inspect the results without running models

Open the [offline results browser](results/review/index.html) to see the inputs, prompts, saved outputs and links to original files. Open [Visual review](additional_analysis/student_review/EDITED_REVIEW.html) for my P02 and P03 notes by job. P03 GLBs can be opened in a compatible 3D viewer; their saved `turntable.mp4` files need no graphics installation.

Read these tables:

- `results/pair_summary.csv`: completion, acceptance and source-group summaries.
- `results/job_scores.csv`: each final job, original strict flags and review notes.
- `results/P01_diagnostics.csv`: supported answer extraction and JSON compliance.
- `results/P04_diagnostics.csv`: strict success and the reason a request failed.
- `results/model_statistics.csv`: measured main-run latency, memory and acceptance.
- `results/resource_usage.csv`: every recorded attempt, including pilots and failures.

P01's historical semantic extractor accepts exact answer JSON directly, inside one conventional JSON fence, or a plain-text branch passed unchanged to the answer matcher. Its policy identifier omits that plain-text branch; none of the 30 final responses uses it. It does not repair invalid JSON. Strict JSON success remains the original recorded flag. P04 retains its exact status/facts/tool contract. Its diagnostic breakdown does not replace the primary score. Blank quality judgments mean pending, not zero.

## Rebuild the analysis on a CPU

Use Python 3.12 on macOS or Linux. From this folder:

```bash
python3.12 -m venv "$HOME/.venvs/comp8410-analysis"
source "$HOME/.venvs/comp8410-analysis/bin/activate"
python -m pip install -r requirements-analysis.txt
python -B scripts/analyse_saved.py
python -B scripts/build_review_index.py
python -B scripts/test_analysis.py
python -B scripts/test_review_index.py
```

The virtual environment is outside this folder. The entry points suppress Python bytecode and store disposable plotting/check caches in the system temporary directory. The CPU analysis does not install Torch, download model weights, contact a model API or perform inference. A separate backup folder is optional. Analysis rewrites derived tables, charts and the feedback summary; it does not rewrite model generations.

The exact CPU validation package versions are recorded in `evidence/environments/analysis_20261103.requirements.txt`. The GPU environment records describe the original Linux runs, not this CPU interpreter. Generated assets and their original texture data remain unchanged.

## Notebook order

1. `prepare_data.ipynb`: inspect the saved job plan, inputs and review worksheet. Leave `DOWNLOAD_PUBLIC_DATA = False` to retain the supplied data.
2. `product_facts_A.ipynb` and `product_facts_B.ipynb`: P01 inputs, settings, optional inference and saved answers.
3. `photo_edit_A.ipynb` and `photo_edit_B.ipynb`: P02 inputs and saved edits.
4. `product_asset_A.ipynb` and `product_asset_B.ipynb`: P03 inputs, shape/texture pipeline and saved assets.
5. `retail_agent_A.ipynb` and `retail_agent_B.ipynb`: P04 requests, tools and saved traces.
6. `score_saved_outputs.ipynb`: reproduce tables and charts.
7. `self_evaluation.ipynb`: summarise the recorded observations.

Install `jupyterlab` if you want its notebook interface. All inference switches are `False` by default. The notebooks find the project when opened from its root or `notebooks/` folder. `python -B scripts/check_notebooks.py` executes inspection copies in a temporary project and checks that no inference/download switches are enabled.

The original GPU execution order was P03, P02, P01 and P04. This resource-based order is different from the order used to explain the tasks.

## Reproduce model inference

Read `GPU_REPRODUCTION.md`. P01/P02/P04 used a shared Python 3.12 CUDA environment; the two P03 pipelines used separate Python 3.10 environments. Exact model IDs, weight revisions, component revisions and scientific settings remain in `config/models.json`. Model weights and installed environments are excluded from the package.

A suitable NVIDIA CUDA system, internet access and any necessary model access terms are required for inference. The saved analysis can be reproduced without those resources. Revised setup instructions were inspected and CPU-tested; no new GPU run was performed during submission preparation.

## Evidence layout

Each job is one input/model/seed combination. `outputs/<pair>/<side>/<phase>/attempts.jsonl` records attempts, and each attempt folder contains settings, outputs and resource samples. The `pilot`, `final` and `reproduction` phases have separate denominators. Keep failed attempts when comparing planned work.

- `config/`: fixed study and model records.
- `data/`: selected source/derived inputs, reference answers, fictional catalogue and review CSV.
- `a1/`: small shared functions for model adapters, workers, scoring and inspection.
- `evidence/environments/`: original installed packages and hardware details.
- `evidence/completion/executed_source/`: source snapshot preserved from execution.
- `evidence/submission_export.json`: original/export checksums for contributor metadata, plus historical source bindings.
- `evidence/pilot_source_history/`: the documented P03 A pilot renderer-source ambiguity.
- `evidence/reproduction/`: repeated-run comparisons and native dependencies.
- `evidence/self_evaluation/`: observations and review records naming the actual jobs inspected.

Historical runs are checked against their recorded source snapshot. A later comment edit does not make the old result appear newly executed. Images, catalogue inputs and generated files must still match their original checksums. Only listed metadata exports can have different explicitly recorded bytes. The original P03 A pilot retains its documented source amendment; it is not silently promoted to strict resume eligibility.

## Data, implementation and licences

ABO images are credited to Amazon.com and the dataset creators under CC BY 4.0. See Collins et al. (2022), `data/provenance.json`, `data/abo_selection.json` and `NOTICE.md`. Source photographs are not IKEA products. Source views and two seeds do not create independent product samples.

I evaluate released model implementations rather than training models or writing their internals. P01 adapts official vision-language interfaces and checks answers against saved labels. P02 calls official image-editing pipelines and evaluates the saved edits. P03 calls the official shape/texture pipelines and uses one shared inspection renderer. P04 supplies the fictional read-only tools and compares saved traces with a deterministic oracle. Original models, dependencies and licence sources are registered in `references/implementation_sources.json`.

## Diagnostics from saved outputs

Read `additional_analysis/README.md`. Run `python -B additional_analysis/rebuild.py` to rebuild only the additional P01/P04 diagnostic CSVs. The label-only P01 shortcut matches 11/15 labels; the separate exact schema validator and all-trace P04 content analysis do not overwrite historical flags or scores. The follow-up protocols describe proposed experiments. The diagnostics use the recorded runs.

The visual review includes the original comments, the incorporated ratings and supporting descriptions of saved views. Read `additional_analysis/student_review/README.txt` for their scope and dates.

## Architecture and visual review

The revised report distinguishes the proposed retail design from the implemented evaluation. `additional_analysis/architecture/` contains the diagram and its editable SVG/JSON source. The four capabilities run independently; the staff portal, shared production catalogue/product-ID mapping and customer-publication links remain proposed.

I reviewed the saved edits and textured drafts. I recorded 108 judgments for 24 jobs, with 93 passes and 15 issues. The dimension judgments and the draft acceptance scores answer different questions. The review page links to the source notes, exact job IDs and saved outputs. The complete comparisons and labelled detail crops are in `additional_analysis/figures_remaining/`.

The organisation scenario uses IKEA as the chosen real organisation and Systems A/B as proposed new capabilities. The supplied lecturer clarification of 4 November 2026 permits this approach; it does not establish that IKEA deploys the evaluated models. Experimental photographs are ABO examples and the tool catalogue is fictional.

The local evaluation folder includes `outputs/`, `results/`, `evidence/` and `data/manual_reviews.csv`. A Git checkout alone would omit these paths under the current ignore rules.
