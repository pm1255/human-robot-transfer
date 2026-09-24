"""Download a bounded public RGB-only HMDB51 view for body-annotation review."""

import json, hashlib
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor


def main():
    root = Path(__file__).resolve().parents[1] / "data/raw/v2_exo"
    root.mkdir(parents=True, exist_ok=True)
    repo = "CVML-TueAI/HMDB51"
    jobs = []
    for label in ["pick", "stand", "walk", "wave", "pour"]:
        rows = json.load(
            urlopen(
                f"https://huggingface.co/api/datasets/{repo}/tree/main/hmdb51_org/{label}?limit=100"
            )
        )
        choices = [
            r
            for r in rows
            if r["path"].endswith(".avi") and 20000 < r.get("size", 0) < 8_000_000
        ][:3]
        for i, row in enumerate(choices):
            jobs.append((label, i, row["path"]))

    def get(job):
        label, i, path = job
        ident = f"hmdb_{label}_{i:02d}"
        dest = root / f"{ident}.avi"
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/" + quote(path)
        if not dest.exists():
            urlretrieve(url, dest)
        return {
            "id": ident,
            "file": dest.name,
            "source_uri": url,
            "source_kind": "human_video",
            "view": "third_person",
            "task": label,
            "task_source": "dataset_action_category",
            "training_eligible": False,
            "purpose": "full-body annotation review; not mixed into bottle-policy pretraining",
            "pose_labels_consumed": False,
            "split_group": path,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }

    with ThreadPoolExecutor(3) as pool:
        records = list(pool.map(get, jobs))
    (root / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    print("downloaded", len(records))


if __name__ == "__main__":
    main()
