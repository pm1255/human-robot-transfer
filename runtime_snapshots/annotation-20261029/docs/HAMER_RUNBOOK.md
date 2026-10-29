# HaMeR 真实推理运行手册

共享目录已经包含：

- 官方源码：`/user/panmiao/human-robot-transfer/vendor/hamer`（commit `3a01849f4148352e9260b69bf28b65d1671a4905`）
- 官方权重：`/user/panmiao/human-robot-transfer/models/hamer/hamer.ckpt`
- 权重 SHA-256：`e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`
- 输入样本：`/user/panmiao/human-robot-transfer/inputs/unlabeled/humanego_serve_bread_000_6s.mp4`
- runner：`human2robot/v2/run_mediapipe_hand.py` 和 `human2robot/v2/run_hamer_mesh.py`

在新申请的 RTX 4090/H100/A100 devspace 中运行：

```bash
cd /user/panmiao/human-robot-transfer
export HAMER_ROOT=$PWD/vendor/hamer
export HAMER_CHECKPOINT=$PWD/models/hamer/hamer.ckpt
export HUMANEGO_INPUT=$PWD/inputs/unlabeled/humanego_serve_bread_000_6s.mp4
export HAND_LANDMARKER_MODEL=$PWD/models/mediapipe/hand_landmarker.task
export HUMANEGO_HAND_OUT=$PWD/outputs/humanego_serve_bread_000_6s/hand
python human2robot/v2/run_mediapipe_hand.py
export HUMANEGO_HAND_JSON=$PWD/outputs/humanego_serve_bread_000_6s/hand/humanego_serve_bread_000_6s_hand.json
export HAMER_OUT_DIR=$PWD/outputs/humanego_serve_bread_000_6s/hamer
python human2robot/v2/run_hamer_mesh.py
```

只有 JSON 中 `vertices_per_hand=778` 且输出视频/JSON均生成后，才能把网页 manifest 中的 HaMeR 状态改为 `actual_hamer`。21点 CPU fallback 不得覆盖该结果。
