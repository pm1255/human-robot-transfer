"""Bounded RGB-only human collection and complete single-task robot view."""

import argparse, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

EPIC = Path(
    "/user/zhangxueqian/dataset/lerobot_v3.0_v0/ego/EpicKitchens/epic_kitchens_action_clips_lda_full_v3"
)
ROBOT = Path(
    "/user/xuwang/dataset_sft_gop2/robotwin-clean-xvla-action-eef/adjust_bottle"
)


def human(out, count=96):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((EPIC / "meta/episodes").rglob("*.parquet"))
        ]
    ).to_dict("records")
    bottle = [
        r for r in rows if "bottle" in str(r.get("vlm_video_instruction", "")).lower()
    ]
    selected = [
        bottle[i]
        for i in np.linspace(0, len(bottle) - 1, min(count, len(bottle)), dtype=int)
    ]
    records = []
    for row in selected:
        name = f"epic_{int(row['episode_index']):06d}"
        p = out / f"{name}.mp4"
        source = EPIC / row["video_path"]
        start = float(row.get("videos/observation.images.head/from_timestamp", 0))
        duration = min(
            12, float(row["videos/observation.images.head/to_timestamp"]) - start
        )
        if not p.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-ss",
                    str(start),
                    "-i",
                    str(source),
                    "-t",
                    str(duration),
                    "-an",
                    "-vf",
                    "scale=640:-2,fps=10",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-movflags",
                    "+faststart",
                    str(p),
                ],
                check=True,
            )
        records.append(
            {
                "id": name,
                "file": p.name,
                "source_uri": str(source),
                "source_cluster": "yingbo_train",
                "task": row.get("vlm_video_instruction", ""),
                "task_source": "upstream_vlm_unverified",
                "view": "egocentric",
                "source_kind": "human_video",
                "pose_labels_consumed": False,
                "training_eligible": True,
                "original_episode": int(row["episode_index"]),
                "split_group": "epic_original_sessions_unresolved",
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "source_start": start,
                "fps": 10,
            }
        )
        print("human", name, flush=True)
    (out / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )


def robot(out, size=128):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((ROBOT / "meta/info.json").read_text())
    fps = info["fps"]
    episodes = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((ROBOT / "meta/episodes").rglob("*.parquet"))
        ]
    )
    order = np.random.default_rng(429).permutation(
        sorted(episodes.episode_index.unique())
    )
    split = {
        int(e): "train" if i < 40 else "validation" if i < 45 else "test"
        for i, e in enumerate(order)
    }
    manifest = []
    for _, r in episodes.sort_values("episode_index").iterrows():
        ep = int(r.episode_index)
        data = ROBOT / info["data_path"].format(
            chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])
        )
        d = pd.read_parquet(data)
        d = d[d.episode_index == ep].sort_values("frame_index")
        n = len(d)
        assert np.array_equal(d.frame_index, np.arange(n))
        assert np.allclose(np.diff(d.timestamp), 1 / fps, atol=1e-4)
        images = []
        for key in ["head_camera", "left_camera", "right_camera"]:
            key = "observation.images." + key
            prefix = "videos/" + key
            video = ROBOT / info["video_path"].format(
                video_key=key,
                chunk_index=int(r[prefix + "/chunk_index"]),
                file_index=int(r[prefix + "/file_index"]),
            )
            raw = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-ss",
                    str(r[prefix + "/from_timestamp"]),
                    "-i",
                    str(video),
                    "-frames:v",
                    str(n),
                    "-vf",
                    f"scale={size}:{size}",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ]
            )
            frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
            assert len(frames) == n
            images.append(frames)
        states = np.stack(d["observation.state"]).astype(np.float32)
        actions = np.stack(d.action).astype(np.float32)
        assert (
            states.shape == (n, 14)
            and actions.shape == (n, 14)
            and np.isfinite(states).all()
            and np.isfinite(actions).all()
        )
        path = out / f"episode_{ep:03d}.npz"
        np.savez_compressed(
            path,
            images=np.stack(images, 1),
            state=states,
            action=actions,
            timestamp=d.timestamp.to_numpy(),
            endpose=np.stack(d["observation.endpose"]).astype(np.float32),
        )
        manifest.append(
            {
                "episode": ep,
                "file": path.name,
                "frames": n,
                "split": split[ep],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print("robot", ep, n, split[ep], flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "task": "adjust_bottle",
                "source": str(ROBOT),
                "fps": fps,
                "cameras": ["head_camera", "left_camera", "right_camera"],
                "state_action": "native absolute 14D joints + grippers, unchanged",
                "episodes": manifest,
                "split_seed": 429,
                "human_hardware_labels_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["human", "robot"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    globals()[a.kind](a.out)
