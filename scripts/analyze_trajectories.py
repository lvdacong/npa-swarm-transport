"""Derive task evidence from saved evaluation trajectories, keeping every request."""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for request in manifest["requests"]:
        row = {key: request.get(key) for key in ("seed", "episode", "layout_seed", "status", "steps")}
        if request.get("trajectory"):
            with np.load(root / request["trajectory"]) as data:
                cargo = data["object_positions"]
                goal = data["goal_position"]
                metrics = json.loads(str(data["metrics_json"]))
                distance = np.linalg.norm(cargo - goal, axis=-1)
                contacts = [m["contact_count"][0] for m in metrics]
                row.update(initial_distance=float(distance[0]), final_distance=float(distance[-1]),
                           minimum_distance=float(distance.min()), progress=float(distance[0] - distance[-1]),
                           displacement=float(np.linalg.norm(cargo[-1] - cargo[0])),
                           maximum_displacement=float(np.linalg.norm(cargo - cargo[0], axis=-1).max()),
                           path_length=float(np.linalg.norm(np.diff(cargo, axis=0), axis=-1).sum()),
                           mean_contacts=float(np.mean(contacts)), maximum_contacts=int(max(contacts)),
                           sampled_fully_connected=not metrics[-1]["ever_disconnected"][0],
                           found_object=metrics[-1]["found_object"][0], found_goal=metrics[-1]["found_goal"][0])
                for key in ("cargo_force_norm", "cargo_force_toward_goal", "reward_contact_credit"):
                    values = [m[key][0] for m in metrics if key in m]
                    if values:
                        row[key + "_mean"] = float(np.mean(values))
                        row[key + "_max"] = float(np.max(values))
        rows.append(row)
    return {"evaluation": str(root.resolve()), "checkpoint": manifest["checkpoint"],
            "requested": manifest["requested"], "diagnostic": manifest["diagnostic"], "episodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "trajectory-analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
