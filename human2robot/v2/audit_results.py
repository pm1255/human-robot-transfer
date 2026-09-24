"""Check matched data/init/budgets before any comparison is published."""

import argparse, json, hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--training", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
root = Path(a.training)
audit = []
manifests = []
for seed in [17, 29]:
    exp = json.loads((root / f"seed{seed}/experiment.json").read_text())
    reports = [
        json.loads((root / f"seed{seed}" / c / "report.json").read_text())
        for c in ["robot_only", "explicit", "implicit_joint", "shuffled"]
    ]
    assert len({r["initial_hash"] for r in reports}) == 1
    assert len({r["steps"] for r in reports}) == 1
    assert len({r["robot_frame_exposures"] for r in reports}) == 1
    assert (
        exp["args"]["pre_steps"] == 2000
        and exp["args"]["fine_steps"] == 8000
        and exp["args"]["action_representation"] == "absolute"
    )
    assert [r["human_frame_exposures"] for r in reports] == [0, 64000, 64000, 64000]
    splits = {
        s: {r["episode"] for r in exp["robot_manifest"]["episodes"] if r["split"] == s}
        for s in ["train", "validation", "test"]
    }
    assert (
        not splits["train"] & splits["test"]
        and not splits["validation"] & splits["test"]
        and not splits["train"] & splits["validation"]
    )
    assert [len(splits[s]) for s in ["train", "validation", "test"]] == [40, 5, 5]
    manifests.append(
        hashlib.sha256(
            json.dumps(exp["robot_manifest"], sort_keys=True).encode()
        ).hexdigest()
    )
    audit.append(
        {
            "seed": seed,
            "same_initialization": reports[0]["initial_hash"],
            "same_robot_frame_exposures": reports[0]["robot_frame_exposures"],
            "steps_per_condition": reports[0]["steps"],
            "human_manifest_sha256": exp["human_manifest_sha256"],
            "split_sizes": [40, 5, 5],
        }
    )
assert len(set(manifests)) == 1
Path(a.out).write_text(
    json.dumps(
        {"passed": True, "audits": audit, "same_robot_manifest_sha256": manifests[0]},
        indent=2,
    )
)
