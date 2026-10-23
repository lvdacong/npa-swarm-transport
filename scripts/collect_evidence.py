"""Index actual local outputs without reclassifying failures as successes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--archive", action="store_true", help="archive reports/trajectories; checkpoints are hash-indexed only")
args = parser.parse_args()
root = args.root.resolve()
if not root.is_dir():
    parser.error("--root must exist")
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
entries = []
archive = zipfile.ZipFile(output / "evidence.zip", "w", zipfile.ZIP_DEFLATED) if args.archive else None
try:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or output in path.parents or path.suffix not in {".json", ".jsonl", ".npz", ".pt", ".png", ".txt", ".log", ".xml"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        if archive is not None and path.suffix != ".pt":
            archive.write(path, relative)
finally:
    if archive is not None:
        archive.close()
(output / "manifest.json").write_text(json.dumps({"kind": "evidence_inventory", "verdict": "not_assigned", "files": entries}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"indexed_files": len(entries), "output": str(output)}))
