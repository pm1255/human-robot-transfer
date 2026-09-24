"""Run all conditions on a frozen expert-feasible seed list."""

import argparse, json, os, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--training-seed", type=int, required=True)
p.add_argument("--training-root", default="training_absolute")
p.add_argument("--evaluation-root", default="evaluation_absolute")
p.add_argument("--conditions", default="robot_only,explicit,implicit_joint,shuffled")
a = p.parse_args()
root = Path(a.root).resolve()
for condition in a.conditions.split(","):
    out = root / a.evaluation_root / f"seed{a.training_seed}" / condition
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        continue
    ckpt = root / a.training_root / f"seed{a.training_seed}" / condition / "final.pt"
    assert ckpt.is_file()
    command = [
        sys.executable,
        "-u",
        "-m",
        "v2.evaluate",
        "--robotwin",
        str(root / "robotwin"),
        "--out",
        str(out),
        "--checkpoint",
        str(ckpt),
        "--seeds",
        str(root / "test_seeds.json"),
        "--single-active-arm",
    ]
    with (out / "rollout.log").open("w") as log:
        subprocess.run(
            command,
            cwd=root / "code",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=4200,
        )
    print("completed", a.training_seed, condition, flush=True)
