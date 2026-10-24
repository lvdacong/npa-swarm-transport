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
                dt = manifest["env_config"]["dt"]
                for key in ("found_object", "found_goal"):
                    first = next((i for i, m in enumerate(metrics) if m[key][0]), None)
                    row[key + "_seconds"] = None if first is None else first * dt
                first = next((i for i, m in enumerate(metrics) if m["found_object"][0]), None)
                if first is not None:
                    after = np.asarray(contacts[first:])
                    longest = current = 0
                    for count in after:
                        current = current + 1 if count > 0 else 0
                        longest = max(longest, current)
                    row.update(post_discovery_contact_fraction=float((after > 0).mean()),
                               post_discovery_maximum_contacts=int(after.max()),
                               longest_continuous_contact_seconds=longest * dt,
                               distinct_contact_agents=int((np.linalg.norm(data["positions"][first:] - cargo[first:, None], axis=-1)
                                   < manifest["env_config"]["object_radius"] + manifest["env_config"]["agent_radius"]).any(0).sum()))
                bridge = np.asarray([m["bridge"][0] for m in metrics], dtype=bool)
                row["bridge_transition_steps"] = (np.flatnonzero(bridge[1:] != bridge[:-1]) + 1).tolist()
                relay = [(i, m) for i, m in enumerate(metrics) if "cargo_near_agents" in m]
                if relay:
                    near = sum(m["cargo_near_agents"][0] for _, m in relay)
                    contact = sum(m["cargo_contact_agents"][0] for _, m in relay)
                    reached = sum(m["goal_relay_near_cargo_agents"][0] for _, m in relay)
                    reached_contact = sum(m["goal_relay_contact_agents"][0] for _, m in relay)
                    row.update(goal_message_near_cargo_agent_fraction=reached / near if near else None,
                               goal_message_contact_agent_fraction=reached_contact / contact if contact else None,
                               contact_decisions=sum(m["cargo_contact_agents"][0] > 0 for _, m in relay),
                               goal_message_contact_decisions=sum(m["goal_relay_contact_agents"][0] > 0 for _, m in relay))
                    for label, key in (("near_cargo", "goal_relay_near_cargo_agents"),
                                       ("contact", "goal_relay_contact_agents")):
                        first = next(((i, m) for i, m in relay if m[key][0] > 0), None)
                        row["first_goal_message_" + label + "_seconds"] = (
                            first[1].get("relay_decision_step", [first[0] - 1])[0] * dt if first else None)
                    if reached:
                        for key in ("goal_bearing_error_near_cargo", "goal_bearing_alignment_near_cargo"):
                            row[key] = sum(m[key][0] * m["goal_relay_near_cargo_agents"][0] for _, m in relay) / reached
                    pushing = [m["cargo_force_toward_goal"][0] for _, m in relay if m["goal_relay_contact_agents"][0] > 0]
                    row["targetward_force_when_contact_has_message"] = float(np.mean(pushing)) if pushing else None
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
