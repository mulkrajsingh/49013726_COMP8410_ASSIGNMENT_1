# Read the selected worker environment and original Hunyuan source folder.
# Build the third-party painter extension needed by that model's setup.
# This is a setup utility, not an evaluation or inference result.

"""Build Hunyuan's painter extension with the selected worker Python.

Run this script from the original Hunyuan repository folder.
"""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import subprocess
import sysconfig
from pathlib import Path


def main():
    import pybind11

    folder = Path.cwd() / "hy3dpaint/DifferentiableRenderer"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("This Python did not provide its extension suffix")
    output = "mesh_inpaint_processor" + suffix
    command = [
        "c++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_paths()["include"],
        "mesh_inpaint_processor.cpp", "-o", output,
    ]
    subprocess.run(command, cwd=folder, check=True)
    print(folder / output)


if __name__ == "__main__":
    main()
