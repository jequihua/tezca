"""Read representative canonical cells and check bundle bytes using only stdlib.

This small consumer demonstrates the CSV contract, not Tezca's full scientific
schema validator. Hashes check integrity, never authenticity or data-use rights.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def read(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "tezca.results/1":
        raise ValueError("unsupported format")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
            raise ValueError("expected a direct result filename")
        path = directory / name
        if path.is_symlink():
            raise ValueError("symlink result refused")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("result checksum mismatch: "+name)
    with (directory / "individual_timeseries_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Canonical cells are typed JSON nodes; they are not ordinary numeric CSV.
    selected = {}
    for row in rows:
        name, status = json.loads(row["result_id"]), json.loads(row["compute_status"])
        if name[0] != "str" or status[0] != "str":
            raise ValueError("expected string result/status cells")
        selected[name[1]] = status[1]
    if any(name == "tezca" or name.startswith("tezca.") for name in sys.modules):
        raise RuntimeError("consumer unexpectedly imported Tezca")
    return dict(bundle_id=manifest["bundle_id"], checked_files=len(manifest["files"]),
                mean_status=selected["signal0"], withheld_band=selected["synoptic"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(read(parser.parse_args().directory)))
