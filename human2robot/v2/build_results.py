"""Merge observed offline and native rollout results, never invent pending scores."""

import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def wilson(k, n):
    if not n:
        return None
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


def build(root, out):
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rollouts = []
    future = []
    for report in sorted(
        (root / "training").glob("seed*/*/report.json"),
        key=lambda p: (
            p.parent.parent.name,
            ["robot_only", "explicit", "implicit_joint", "shuffled"].index(
                p.parent.name
            ),
        ),
    ):
        condition = report.parent.name
        seed = report.parent.parent.name
        name = f"{condition} / {seed}"
        r = json.loads(report.read_text())
        row = {
            "name": name,
            "offline": r["test"],
            "pretrain_offline": json.loads(
                (report.parent / "pretrain_offline.json").read_text()
            ),
            "closed_loop": None,
            "ci": None,
            "training": {k: v for k, v in r.items() if k != "test"},
        }
        d = np.load(report.parent / "final_predictions.npz")
        samples = []
        for i in range(len(d["episode"])):
            samples.append(
                {
                    k: d[k][i].tolist()
                    for k in [
                        "predicted",
                        "target",
                        "state",
                        "mask",
                        "episode",
                        "frame",
                    ]
                }
            )
        filename = f"{seed}_{condition}_predictions.json"
        (out / filename).write_text(
            json.dumps({"samples": samples}, separators=(",", ":"))
        )
        row["predictions"] = "v2data/" + filename
        ev = root / "evaluation" / seed / condition
        if (ev / "summary.json").exists():
            summary = json.loads((ev / "summary.json").read_text())
            row["closed_loop"] = summary
            row["ci"] = wilson(summary["successes"], summary["valid"])
            for rec in summary["records"]:
                file = f'{seed}_{condition}_{rec["seed"]}.mp4'
                if (out / "rollouts" / file).exists():
                    rec = {**rec, "condition": name, "video": "v2data/rollouts/" + file}
                    tracepath = ev / f'actions_{rec["seed"]}.npz'
                    if tracepath.exists():
                        trace = np.load(tracepath)
                        fn = f'{seed}_{condition}_{rec["seed"]}_trace.json'
                        (out / fn).write_text(
                            json.dumps(
                                {
                                    k: trace[k].tolist()
                                    for k in [
                                        "state",
                                        "action",
                                        "raw_action",
                                        "physical_before",
                                        "physical_after",
                                        "bottle_point",
                                    ]
                                    if k in trace
                                },
                                separators=(",", ":"),
                            )
                        )
                        rec["trace"] = "v2data/" + fn
                    rollouts.append(rec)
        rows.append(row)
        imagefile = report.parent / "pretrain_future_images.npz"
        if imagefile.exists():
            d = np.load(imagefile)
            canvas = Image.new("RGB", (384, 160 * min(6, len(d["current"]))), "white")
            draw = ImageDraw.Draw(canvas)
            for i in range(min(6, len(d["current"]))):
                for col, key in enumerate(["current", "predicted", "future"]):
                    canvas.paste(
                        Image.fromarray(d[key][i]).resize((128, 128)),
                        (col * 128, i * 160 + 24),
                    )
                    draw.text(
                        (col * 128 + 5, i * 160 + 5),
                        ["Current", "Predicted +0.5s", "Observed +0.5s"][col],
                        fill="black",
                    )
            namefile = f"{seed}_future_images.jpg"
            canvas.save(out / namefile)
            future.append({"seed": seed, "image": "v2data/" + namefile})
    result = {
        "protocol": "adjust_bottle · 50 episodes → 40 train / 5 validation / 5 test · 3 相机 · 16 步动作序列 · 每条件 2,000 步联合预训练 + 8,000 步相同机器人微调 · 绝对关节目标 · seeds 17 / 29",
        "controller_note": "主结果使用绝对关节目标。增量版本在独立开发场景出现累计漂移，保留失败录像用于审计，不并入最终成功率。",
        "rows": rows,
        "rollouts": rollouts,
        "future_images": future,
        "interpretation": "主要结论以同一场景的原生闭环成功率为准。离线误差不等于成功率；图像、运动和动作损失不跨路线排名。样本量有限，两个训练种子不能证明普遍替代真机数据。",
    }
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.root, a.out)
