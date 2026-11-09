# Read existing inputs, attempts and outputs; display them in a notebook.
# Show the saved evidence without model inference or new quality judgments.

"""Show existing model evidence in notebooks without running or judging a model."""

import json
import os
from pathlib import Path

from .io import project_path, read_attempts, read_json
from .study import planned_jobs


def show_saved_results(root, pair, side):
    """Display inputs, attempt history and saved outputs for one model."""
    from IPython.display import FileLink, Image, Markdown, display

    root = Path(root).resolve()
    print("Saved evidence inspection only. Statuses come from the logs; no model is run or scored.")

    def show_file(path, image=False, text=False):
        if not path.is_file():
            print("Pending or missing:", path.relative_to(root))
            return
        display(FileLink(os.path.relpath(path, Path.cwd())))
        if image:
            try:
                display(Image(filename=str(path), width=360))
            except (OSError, ValueError) as error:
                print("Cannot display saved image:", error)
        if text:
            try:
                content = path.read_text()
                try:
                    content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
                except ValueError:
                    pass
                print(content)
            except (OSError, UnicodeError) as error:
                print("Cannot read saved text:", error)

    for phase in ["pilot", "final", "reproduction"]:
        jobs = {job["job_id"]: job for job in planned_jobs(root, pair, phase)
                if job["side"] == side}
        log = root / f"outputs/{pair}/{side}/{phase}/attempts.jsonl"
        try:
            rows = read_attempts(log)
        except (OSError, ValueError) as error:
            print("Cannot inspect attempt log:", error)
            continue
        histories = {}
        for row in rows:
            histories.setdefault(row["job_id"], []).append(row)
        if rows:
            print("Full attempt log for", phase)
            show_file(log)
        # Retain jobs from earlier settings as well as the current plan.
        ids = list(jobs) + [job_id for job_id in histories if job_id not in jobs]
        for job_id in ids:
            display(Markdown(f"### {phase}: {job_id}"))
            history = histories.get(job_id, [])
            latest = history[-1] if history else None
            item = jobs.get(job_id, {}).get("item", {})
            folder = None
            if latest is None:
                print("Pending: no saved attempt for this planned job.")
            else:
                attempts = {}
                for row in history:
                    attempts[row["attempt"]] = row
                for number, row in attempts.items():
                    print("Attempt", number, "recorded status:", row["status"], row.get("error", ""))
                if job_id not in jobs:
                    print("This saved job uses earlier settings and is outside the current plan.")
                relative = latest.get("partial_directory")
                if latest.get("output_path"):
                    relative = str(Path(latest["output_path"]).parent)
                if relative:
                    folder = project_path(root, relative)
                else:
                    folder = log.parent / job_id / f"attempt_{latest['attempt']:02d}"
                    if not folder.is_dir():
                        folder = folder.with_name(folder.name + ".partial")
                settings = folder / "settings.json"
                if settings.is_file():
                    try:
                        item = read_json(settings)["job"]["item"]
                        print("Input below is taken from this saved attempt's settings.")
                        show_file(settings)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        print("Cannot read saved settings:", error)
                else:
                    print("Saved settings are missing; any input below is from the current plan.")
            if item.get("prompt"):
                print("Prompt:", item["prompt"])
            if item.get("image"):
                show_file(project_path(root, item["image"]), image=True)
            if item.get("catalogue"):
                print("Mock catalogue input:")
                show_file(project_path(root, item["catalogue"]))
            if folder is None:
                continue
            print("Files from the latest saved attempt:")
            if pair == "P01":
                show_file(folder / "answer.json", text=True)
            elif pair == "P02":
                show_file(folder / "image.png", image=True)
            elif pair == "P03":
                show_file(folder / "preview.png", image=True)
                show_file(folder / "turntable.mp4")
                show_file(folder / "asset.glb")
            elif pair == "P04":
                show_file(folder / "agent.json", text=True)
            if (folder / "failure.json").is_file():
                show_file(folder / "failure.json", text=True)
