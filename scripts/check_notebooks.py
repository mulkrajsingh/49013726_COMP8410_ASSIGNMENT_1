# Read all eleven notebooks; run CPU inspection copies in a temporary project.
# Require inference and public-data download switches to remain false.
# Submitted cells, outputs and experiment evidence are not written by this check.

"""Run CPU inspection copies of all eleven notebooks; never enable inference."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import ast
import json
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


# Keep original generations linked read-only in practice, while making separate
# small metadata copies. Analysis writes only inside the temporary project.
def copy_file(source, target):
    if Path(source).stat().st_size > 1024 * 1024:
        try:
            os.link(source, target)
            return target
        except OSError:
            pass
    return shutil.copy2(source, target)


def main():
    with tempfile.TemporaryDirectory(prefix="comp8410_notebooks_") as temporary:
        work = Path(temporary)
        project = work / "Evaluation"
        for folder in ["a1", "config", "data", "outputs", "evidence", "notebooks", "results"]:
            shutil.copytree(ROOT / folder, project / folder, copy_function=copy_file,
                            ignore=shutil.ignore_patterns("__pycache__", "*.whl", "*.so", "*.pt", "*.obj"))
        # The validator needs every recorded artifact, including native intermediates.
        # Link those files too, without copying gigabytes or loading a model.
        for folder in ["outputs"]:
            for source in (ROOT / folder).rglob("*"):
                if source.is_file() and source.suffix in {".pt", ".obj"}:
                    target = project / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copy_file(source, target)
        os.environ["IPYTHONDIR"] = str(work / "ipython")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "jupyter")
        os.environ["MPLCONFIGDIR"] = str(work / "matplotlib")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        specs = work / "kernels/python3"
        specs.mkdir(parents=True)
        (specs / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "CPU analysis", "language": "python"}))
        manager = KernelSpecManager(kernel_dirs=[str(specs.parent)], ensure_native_kernel=False)
        notebooks = sorted((project / "notebooks").glob("*.ipynb"))
        if len(notebooks) != 11:
            raise ValueError("Expected eleven submitted notebooks")
        checked = []
        for path in notebooks:
            book = nbformat.read(path, as_version=4)
            for cell in book.cells:
                if cell.cell_type != "code":
                    continue
                for node in ast.walk(ast.parse(cell.source)):
                    if isinstance(node, ast.Assign):
                        for name in node.targets:
                            if isinstance(name, ast.Name) and (name.id.startswith("EXECUTE_") or name.id == "DOWNLOAD_PUBLIC_DATA"):
                                if not isinstance(node.value, ast.Constant) or node.value.value is not False:
                                    raise ValueError("Unsafe inspection switch in " + path.name)
            kernel = KernelManager(kernel_name="python3", kernel_spec_manager=manager)
            try:
                NotebookClient(book, km=kernel, timeout=240,
                               resources={"metadata": {"path": str(project / "notebooks")}}).execute()
            finally:
                if kernel.has_kernel:
                    kernel.shutdown_kernel(now=True)
                kernel.cleanup_resources()
            checked.append(path.name)
            print("Passed CPU inspection:", path.name, flush=True)
        print(json.dumps({"notebooks_passed": checked, "count": len(checked), "model_inference": False, "model_download": False}))


if __name__ == "__main__":
    main()
