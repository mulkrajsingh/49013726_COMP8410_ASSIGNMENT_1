# Read the fixed source selection; download only its registered photographs.
# Check source hashes before accepting the downloaded files.
# This optional preparation step is separate from saved-evidence analysis.

"""Download only the original source photographs used by the fixed study."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.data import download
from a1.io import atomic_json, read_json, utc_now


def selected_files(root):
    # The provenance record identifies the exact photographs that produced the
    # submitted inputs. Extra catalogue indexes and unused rotations are omitted.
    records = read_json(Path(root) / "data/provenance.json")["records"]
    return [{"url": row["url"], "path": row["path"], "sha256": row["sha256"]}
            for row in records if row.get("url")]


def collect(root):
    root = Path(root).resolve()
    files = selected_files(root)
    receipt = {"started_at": utc_now(), "complete": False, "planned_files": len(files),
               "scope": "source photographs used by the fixed study; no model runs",
               "files": [], "errors": []}
    output = root / "evidence/data_download.json"
    atomic_json(output, receipt)

    def fetch(item):
        record = download(item["url"], root / item["path"])
        if item.get("sha256") and record["sha256"] != item["sha256"]:
            raise ValueError(f"Source metadata checksum changed: {item['path']}")
        record["path"] = item["path"]
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, item): item for item in files}
        for future in concurrent.futures.as_completed(pending):
            try:
                receipt["files"].append(future.result())
            except Exception as error:
                receipt["errors"].append({"path": pending[future]["path"], "error": str(error)})
            atomic_json(output, receipt)
            done = len(receipt["files"]) + len(receipt["errors"])
            if done % 25 == 0:
                print(f"Checked/downloaded {done}/{len(files)} files", flush=True)
    receipt["files"].sort(key=lambda row: row["path"])
    receipt["finished_at"] = utc_now()
    receipt["complete"] = not receipt["errors"]
    receipt["total_bytes"] = sum(row["bytes"] for row in receipt["files"])
    atomic_json(output, receipt)
    print(f"Saved {len(receipt['files'])} source files; {len(receipt['errors'])} errors", flush=True)
    if receipt["errors"]:
        raise SystemExit("Some source files need attention; see evidence/data_download.json")


if __name__ == "__main__":
    collect(ROOT)
