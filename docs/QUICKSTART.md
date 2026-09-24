# 从全新克隆开始

统一入口是仓库根目录的 `tools/run.py`，不要直接启动 `runtime_snapshots/` 内的集群脚本。后者是历史溯源记录，包含原实验路径。当前入口运行短片标注/推理，不训练模型。

## 环境

使用 Python 3.11 或 3.12，推荐 Ubuntu Linux。

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-venv libgl1 libglib2.0-0
git clone https://github.com/pm1255/human-robot-transfer.git
cd human-robot-transfer
bash scripts/setup.sh cpu
source .venv/bin/activate
mkdir -p inputs outputs
python tools/run.py --help
```

macOS 可先执行 `brew install ffmpeg`，但部分无图形上下文环境中的 MediaPipe 无法初始化，即使指定 CPU 也可能失败；此时使用 Linux。几何渲染安装 `bash scripts/setup.sh render`；脚本不会启动仿真器或控制真机。

模型下载地址来自 [MediaPipe Hand](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index)、[Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index) 和官方 EfficientDet Lite0。默认下载到 `models/mediapipe/`；也可 `python tools/run.py models --models /path/to/models`。联网失败时重新执行下载命令，完成的模型保留，`.part` 文件不当成有效权重。

## 输入和输出

- `annotate`：用户视频 + `--view egocentric` 或 `exocentric`，输出可播放的标注叠加、JSON 和 NPZ。
- `arm`：用户手部视频 + Aloha 资产，输出人手/掩码/机器人三栏预览及审计。
- `humanoid`：第三人称全身视频 + G1 29 DOF XML 和网格，输出全身标注及三栏预览。
- `robot2human`：机器人视频 + 同步黑白掩码视频 + 人手参考图片 + 描述 + Wan 源码/权重，输出合成、模型原始输出及掩码对照。

视频需用户自行准备；README 中的动画是压缩预览，不是原始输入。每次选择新的输出目录；已有非空目录会报错，避免用新视频搭配旧缓存。标注/几何预览默认截取前 6 秒、10 FPS，最多 12 秒；VACE 最多 6 秒，默认重采样到 5 FPS、256² 后生成 512² 视频，不能因此恢复缺失细节。

## Aloha 资产

资产不随此仓库重新授权。按 [RoboTwin 官方 Install & Download](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 下载 embodiments。将 `--aloha-assets` 指向其中的 `aloha-agilex` 模型目录，而不是机器人数据集或 checkpoint 目录。

入口要求目录至少包含 `model.urdf`，以及 URDF 使用的 `base_arm.dae`、`link1.dae` … `link8.dae` 和纹理（例如 `Image_269.png`）。网格可位于子目录，但同名文件不得冲突；启动时复制到输出工作目录，再生成 `arm.urdf`，不会改写源资产。仅下载 RoboTwin 源代码通常不包含全部资产，必须完成官方资产下载步骤。

只下载机器人资产（不下载物体和背景包）的初始命令：

```bash
source .venv/bin/activate
python -m pip install huggingface_hub
mkdir -p third_party/robotwin-assets
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TianxingChen/RoboTwin2.0', repo_type='dataset', filename='embodiments.zip', local_dir='third_party/robotwin-assets')"
python -m zipfile -e third_party/robotwin-assets/embodiments.zip third_party/robotwin-assets
```

下载源与 RoboTwin 官方 `assets/_download.py` 一致。解压后的 Aloha 目录通常是 `third_party/robotwin-assets/embodiments/aloha-agilex`；若上游压缩包调整了顶层目录，用 `find third_party/robotwin-assets -name model.urdf` 定位其中的 `aloha-agilex`，将实际路径传给 `--aloha-assets`。

```bash
bash scripts/setup.sh render
source .venv/bin/activate
python tools/run.py arm --video inputs/human.mp4 \
  --aloha-assets /path/to/aloha-agilex --out outputs/arm
```

## G1 资产

```bash
bash scripts/setup.sh render
source .venv/bin/activate
mkdir -p third_party
git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
python tools/run.py humanoid --video inputs/full_body.mp4 \
  --g1-xml third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml \
  --out outputs/humanoid
```

保留 XML 周围的 meshes 目录。渲染器针对 G1 29 DOF 的关节数量和命名；不支持任意 MJCF 自动重定向。无需启动 Unitree 的控制程序或连接真机。

## VACE 环境

使用独立环境避免 CUDA 视频模型与标注依赖混装。安装前确认 NVIDIA 驱动正常，运行 `nvidia-smi`。模型下载需要较大磁盘空间，网络/下载耗时不计入“一键推理”。原实验在 H100 上运行，1.3B 的显存峰值约 16 GiB，建议留出额外余量；此值不能保证其他 CUDA/PyTorch 组合也相同。

```bash
# 仓库根目录；安装上游依赖并下载 VACE 1.3B
bash scripts/setup_vace.sh
source .venv-vace/bin/activate

python tools/run.py robot2human --video inputs/robot.mp4 \
  --mask inputs/robot_mask.mp4 --reference inputs/human_reference.png \
  --prompt "A human hand grasps the cup and lifts it with a stable wrist." \
  --wan third_party/Wan2.1 --weights models/Wan2.1-VACE-1.3B \
  --out outputs/robot2human
```

`setup_vace.sh` 使用官方 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 代码与 `Wan-AI/Wan2.1-VACE-1.3B` 权重。默认 PyTorch wheel 不匹配驱动时，先按 PyTorch 官方方式在 `.venv-vace` 中安装匹配版本，再继续脚本。本项目的 SDPA 路径允许不安装 flash-attn。

掩码与源视频必须从同一时刻开始、保持同一时间轴。白色是需要替换的机器人区域，黑色是保留区域；可用 SAM2 得到掩码后人工检查。入口验证长度和重采样后的帧数，但不自动验证分割质量。参考图应体现目标人手和相容的视角。旧 `segment_expanded.py` 中的坐标只适用于已展示的 6 个场景，不适用于任意新视频。

## 复现范围

“一键启动”指前置资源准备后，用一条命令串联已有处理阶段；不等于自动获得许可、提供原数据或保证转换质量。新启动入口的验证记录见 [VALIDATION.md](../VALIDATION.md)。完整 GPU 依赖安装和新自定义视频的生成结果不得在未运行时宣称已验证。
