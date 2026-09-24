"""Download a bounded public RGB-only HMDB51 view for body-annotation review."""

import json, hashlib, numpy as np
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor


def main():
    root = Path(__file__).resolve().parents[1] / "data/raw/v2_body"
    root.mkdir(parents=True, exist_ok=True)
    repo = "CVML-TueAI/HMDB51"
    jobs = []
    for label in ["golf", "swing_baseball", "kick_ball", "run", "cartwheel"]:
        rows = json.load(
            urlopen(
                f"https://huggingface.co/api/datasets/{repo}/tree/main/hmdb51_org/{label}?limit=100"
            )
        )
        choices = [
            r
            for r in rows
            if r["path"].endswith(".avi") and 20000 < r.get("size", 0) < 8_000_000
        ]
        choices = [choices[i] for i in np.linspace(0, len(choices) - 1, 4, dtype=int)]
        for i, row in enumerate(choices):
            jobs.append((label, i, row["path"], row["size"]))

    def get(job):
        label, i, path, size = job
        ident = f"hmdb_body_{label}_{i:02d}"
        dest = root / f"{ident}.avi"
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/" + quote(path)
        if not dest.exists() or dest.stat().st_size != size:
            for attempt in range(4):
                try:
                    urlretrieve(url, dest.with_suffix(".part"))
                    assert dest.with_suffix(".part").stat().st_size == size
                    dest.with_suffix(".part").replace(dest)
                    break
                except Exception:
                    if attempt == 3:
                        raise
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
