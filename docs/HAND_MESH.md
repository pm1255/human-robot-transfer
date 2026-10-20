# HaMeR 手型估计与显示修复

新增的 `human2robot/hand_mesh` 从真实 HaMeR checkpoint 产生每只手 778 个顶点和 MANO 姿态参数，用原始 1538 个三角面渲染完整表面。手骨架和离散手型不作为三维网格替代品。结果仍是单目估计，不能仅凭顶点数将其视为三维真值或机器人动作。

## 项目位置与版本

- 本仓库是 `pm1255/human-robot-transfer` 的完整代码；本次开发基于 `29c663509d057b7c77b7b9b3b64f6ca93920c232`。
- 服务器运行目录：`/user/panmiao/human-robot-transfer`。该目录还包含 GitHub 未收录的后续标注脚本、模型、输入和输出。
- 新增运行入口：`tools/run_hamer_surface_job.sh`；模型代码由 `--hamer-root` 指向已有官方 HaMeR checkout。
- 数据：EgoVerse `human_bimanual`、`EgoWAM_fold_clothes`、`2026-04-28-21-04-25-000000`，640×480、30 FPS、4834 帧。

## 修复范围

1. 旧 `run_hamer_mesh_fast.py` 在 `HAMER_FAST_RENDER=1` 时只画散点和凸包。新渲染器逐三角面进行透视插值、深度比较和顶点法线光照，保留指缝与自遮挡。
2. 旧裁剪先扩边、裁到图像边界，再放大 2 倍。新入口直接由 21 点构造边界框，只在官方 `ViTDetDataset` 内放大一次，边缘手保留原始中心。
3. 左手顶点仅反射一次，三角面绕序反转一次；相机平移只加一次。相机约定为 OpenCV：x 向右、y 向下、z 向前。
4. 流式解码和 H.264 输出，检查解码/编码返回码、帧数和尺寸。预测压缩分块保存，异常不被吞掉后当作成功。
5. 保存 3D 关节、MANO 参数、模型/输入 SHA-256、相机约定、逐帧缺失与二维预测间偏差。提供 TensorBoard 推理/渲染/验证进度和抽帧图像。
6. 原检测在 570 帧出现同一帧两只手都为同一侧。新增跨帧全局分配，对手腕和掌部运动连续性约束，使左右身份在单帧内互斥；保留原始检测类别与置信度，改动处重算 HaMeR，不只修改颜色。

## 使用

先准备官方 HaMeR 依赖、已有 checkpoint、MANO_RIGHT.pkl 和完整视频对应的 MediaPipe 检测 JSON。运行位置应能解析 checkpoint 配置中的 MANO 相对路径。

```bash
export PYTHONPATH=/path/to/human-robot-transfer:/path/to/hamer:$PYTHONPATH
python -m human2robot.hand_mesh.identity \
  --input sample_hand.json --output sample_hand_tracked.json
python -m human2robot.hand_mesh.infer \
  --video sample.mp4 --hands sample_hand_tracked.json \
  --checkpoint /path/to/hamer_ckpts/checkpoints/hamer.ckpt \
  --hamer-root /path/to/hamer --out /new/output/directory
```

任意兼容 CUDA GPU 均可；也支持 `--device cpu --threads 8`。支持 AMX 的 CPU 可加 `--cpu-bf16`，但应先与 FP32 对照；本样本前 64 帧检查中，关节投影差中位数 0.41 px、P95 0.87 px。MANO 几何保留 FP32。CPU 仅重绘已有顶点时无需加载模型，使用 `python -m human2robot.hand_mesh.replay --help`。渲染额外依赖 NumPy、Numba、OpenCV、FFmpeg；监控依赖 TensorBoard。

输出 `hamer_surface.mp4`（原视频叠加）、`hamer_mesh_only.mp4`（同一相机下纯网格）、`predictions/*.npz`、`faces.npy`、`camera.npz`、`frames.jsonl`、`manifest.json`。NPZ 中 MANO 旋转仍为规范右手参数，`is_right` 指示最终反射；vertices/joints 已转换为原始图像左右手。

## 质检边界

- `CONSISTENT_2D` 表示与 MediaPipe 预测的二维点大致一致，不是人工真值精度通过。
- 当中位关键点偏差超过手框对角线 20%，或超过半数顶点出框时标记 `REVIEW`。20% 是用于定位问题的工程阈值，尚未用人工金标集标定。
- 无检测帧保留缺失，不以插值冒充模型观察。遮挡手指是模型预测。
- 深度只处理手自身及双手相互遮挡；场景/衣物深度未知，网格叠加可能覆盖实际遮挡物。
- 焦距来自 HaMeR 模型约定，非相机实测标定；三维尺度、接触和跨帧身份没有独立真值验证。

回归验证：`PYTHONPATH=. python tests/hand_mesh/test_render.py`，覆盖投影/左右反射、三角面指缝、深度顺序和“778 顶点不等于质量通过”。

网页视频应通过 `python tools/serve_hand_review.py --directory /path/to/results` 提供服务。该入口支持 HTTP Range，可随机拖动 MP4；终端打印本机入口。不要把修复视频发布到原公开站点，除非用户要求发布。
