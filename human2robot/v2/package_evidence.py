"""Build local review assets without publishing source material."""

import json, subprocess, shutil, tarfile, argparse
from pathlib import Path
from v2.review_gate import gate

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
a = p.parse_args()
root = Path(a.root)
out = root / "evidence_web"
out.mkdir(exist_ok=True)
(out / "media").mkdir(exist_ok=True)
(out / "annotations").mkdir(exist_ok=True)
manifest = []
for group, raw in [("human", "human"), ("exo", "v2_exo"), ("body", "v2_body")]:
    for path in sorted((root / "annotations" / group).glob("*/summary.json")):
        s = json.loads(path.read_text())
        ident = s["id"]
        source = root / "data" / raw / s["file"]
        dest = out / "media" / f"{ident}.mp4"
        if not dest.exists():
            if source.suffix == ".mp4":
                shutil.copy2(source, dest)
            else:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-threads",
                        "1",
                        "-i",
                        str(source),
                        "-t",
                        "12",
                        "-vf",
                        "scale=640:-2,fps=10",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-crf",
                        "24",
                        "-movflags",
                        "+faststart",
                        str(dest),
                    ],
                    check=True,
                )
        qualified = gate(json.loads((path.parent / "annotation.json").read_text()))
        s = qualified["meta"]
        (out / "annotations" / f"{ident}.json").write_text(
            json.dumps(qualified, separators=(",", ":"))
        )
        records = qualified["frames"]
        best = max(
            records,
            key=lambda r: len(r["hands"]) * 3
            + int(r["full_body_visible"]) * 7
            + len([o for o in r["objects"] if o["target_candidate"]]),
        )["t"]
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(best),
                "-i",
                str(dest),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2",
                "-threads",
                "1",
                str(out / "media" / f"{ident}.jpg"),
            ],
            check=True,
        )
        manifest.append(s)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("packaging", len(manifest), flush=True)
with tarfile.open(root / "evidence_web.tgz", "w:gz") as tar:
    tar.add(out, arcname="v2data")
