"""Return all metrics/actions plus explicitly selected representative videos."""

import argparse, json, tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
a = p.parse_args()
root = Path(a.root)
with tarfile.open(root / "evaluation_reports.tgz", "w:gz") as tar:
    for folder in sorted((root / "evaluation_absolute").glob("seed*/*")):
        for f in folder.iterdir():
            if f.suffix in [".json", ".npz"]:
                tar.add(
                    f,
                    arcname=str(
                        Path("evaluation") / f.relative_to(root / "evaluation_absolute")
                    ),
                )
        summary = folder / "summary.json"
        if not summary.exists():
            continue
        rows = json.loads(summary.read_text())["records"]
        selected = []
        for success in [True, False]:
            match = next(
                (r for r in rows if r["valid"] and r["success"] == success), None
            )
            if match:
                selected.append(match["seed"])
        for r in rows:
            if len(selected) >= 3:
                break
            if r["seed"] not in selected:
                selected.append(r["seed"])
        for seed in selected:
            f = folder / f"rollout_{seed}.mp4"
            if f.exists():
                tar.add(
                    f,
                    arcname=f"web_rollouts/{folder.parent.name}_{folder.name}_{seed}.mp4",
                )
    for f in [
        "test_seeds.json",
        "expert_seeds/summary.json",
        "expert_seeds/valid_seeds.json",
    ]:
        tar.add(root / f, arcname=f)
print("packaged", root / "evaluation_reports.tgz")
