# Read existing attempts, reviews and labels; rebuild CPU tables and figures.
# This entry point writes derived analysis files, not new model outputs.

"""Rebuild the submitted tables and figures from saved outputs on a CPU."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a1.analysis import plot_results, rebuild_results
from a1.diagnostics import rebuild_diagnostics, rebuild_model_statistics
from a1.feedback import summarize_feedback


# This entry point never loads a model. It reads the supplied attempts and
# review worksheet, then writes only the derived result and feedback summaries.
if __name__ == "__main__":
    print(rebuild_results(ROOT))
    print(rebuild_diagnostics(ROOT))
    print(rebuild_model_statistics(ROOT))
    print(summarize_feedback(ROOT))
    print(plot_results(ROOT))
