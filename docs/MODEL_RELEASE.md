# 模型、配套脚本与复现范围

**这些模型覆盖现有手部检测、手型估计、视频分割、单目深度与物体位姿试验；它们不构成一个端到端“数据闭环模型”。** 运行还需要源码、依赖环境、输入数据和配置。语义段来自已有任务说明及人工/助手逐帧整理；二维接触与质量分来自规则和一致性计算，本项目没有额外训练好的通用语义分段、物理接触、可供性或抓取预测权重。

- 配套源码：https://github.com/pm1255/human-robot-transfer
- 模型仓库：https://huggingface.co/erwtwert/embodied-data-loop-models
- 逐文件大小、SHA-256、许可、安装路径：[manifest.json](../model_assets/manifest.json)
- 模型仓库说明的维护来源：[MODEL_CARD.md](../model_assets/MODEL_CARD.md)

清单记录需要哪些文件；Hub 文件是否已发布必须检查实际文件列表或运行检查工具。公开仓库不意味着所有模型都适用 MIT 或都可商用。

## 模型与脚本对应表

| 功能 | 权重与配套资产 | GitHub 实现 / 入口 | 范围与限制 |
|---|---|---|---|
| 手、身体关键点和基础物体检测 | MediaPipe Hand Landmarker、Pose Landmarker Full、EfficientDet Lite0 | [`tools/run.py`](../tools/run.py)、[`human2robot/v2/run_mediapipe_hand.py`](../human2robot/v2/run_mediapipe_hand.py) | 检测可能漏检；离散手势是几何规则 |
| 三维手型 | HaMeR `hamer.ckpt`、`model_config.yaml`、`mano_mean_params.npz`，另需自行获取 `MANO_RIGHT.pkl` | [`human2robot/hand_mesh/infer.py`](../human2robot/hand_mesh/infer.py)、[`identity.py`](../human2robot/hand_mesh/identity.py)、[`render.py`](../human2robot/hand_mesh/render.py) | 单目估计；MANO 资产不由本仓库重分发 |
| 视频实例分割、物体框 | SAM 3.1 `sam3.1_multiplex.pt` 与配置/分词文件 | [`sam3_video.py`](../human2robot/object_annotation/sam3_video.py)、[`unified_multimodal.py`](../human2robot/object_annotation/unified_multimodal.py) | 框由预测掩码派生；空间提示需按实际输入准备 |
| 单目深度 | MoGe-3 ViT-L `model.pt` | [`depth_video.py`](../human2robot/object_annotation/depth_video.py) | 当前固定校验模型版本；需要 CUDA/FlexGEMM，非实测深度 |
| 物体位姿试验 | FoundationPose refiner、scorer，各自的 `config.yml` | [`foundation_cloth.py`](../human2robot/object_annotation/foundation_cloth.py) | 柔性衣物只输出刚体代理；非商业研究/评估许可 |
| 自动质检 | 不需独立训练权重；部分输入依赖上述模型 | [`confidence.py`](../human2robot/object_annotation/confidence.py)、[`audit_multimodal.py`](../human2robot/object_annotation/audit_multimodal.py)、[`batch30_validate.py`](../local_annotation_lab/unified/batch30_validate.py) | 分数用于筛查，未经概率校准 |
| 统一展示与批处理 | 上述模型的派生结果、JSON 和视频 | [`local_annotation_lab/`](../local_annotation_lab/)、[`local_annotation_lab/unified/`](../local_annotation_lab/unified/)、[`serve_hand_review.py`](../tools/serve_hand_review.py) | 网页源码不包含原始数据和视频 |
| 可选：机器人转人视频 | 官方 Wan2.1-VACE-1.3B 完整快照（含 VAE、文本编码器、分词器） | [`scripts/setup_vace.sh`](../scripts/setup_vace.sh)、[`tools/run.py`](../tools/run.py) 的 `robot2human` | 从官方 Hub 获取；非核心标注依赖 |
| 可选：机器人几何替换 | Aloha/G1 的 URDF、XML 和网格 | [`docs/QUICKSTART.md`](QUICKSTART.md)、[`human2robot/retarget_cloth/`](../human2robot/retarget_cloth/) | 机器人资产需另行准备；预览不等于动作可执行 |

## 获取与检查

在仓库根目录执行：

```bash
python -m pip install huggingface_hub
python tools/model_assets.py list

# 按需下载；只有文件确实发布到 Hub 且当前账户有访问权限时才会成功。
python tools/model_assets.py download --component mediapipe --component moge3
python tools/model_assets.py download --component hamer --component sam31 --component foundationpose

# 检查所有核心文件。缺文件、字节数不符、SHA-256 不符会分别报告并返回非零值。
python tools/model_assets.py check --report outputs/model_assets_check.json
```

`--root /path/to/checkout` 可指定安装根目录；`--revision <Hub commit>` 可固定模型仓库版本。工具不会自动下载 MANO。取得 MANO 授权后，将文件放在 `models/hamer/_DATA/data/mano/MANO_RIGHT.pkl`。清单校验值对应本轮实际运行版本；其他合法版本会显示哈希不匹配，需要重新验证兼容性。

模型数据合计约 **7.97 GB**（十进制，含分词和配置文件，不含 MANO、VACE、机器人资产与依赖环境）。HaMeR 的两个服务器副本内容相同，清单只保留一份。单个 `.pt` 文件不能替代需要的配置文件。

## 源码依赖

建议 Linux + NVIDIA CUDA。基础预览按 [`QUICKSTART.md`](QUICKSTART.md) 安装。高级模块需分别遵循对应官方仓库的环境说明：

```bash
git clone https://github.com/geopavlakos/hamer.git vendor/hamer
git clone https://github.com/facebookresearch/sam3.git vendor/sam3-main
git clone https://github.com/microsoft/MoGe.git vendor/MoGe-main
git clone https://github.com/NVlabs/FoundationPose.git vendor/FoundationPose-main
```

上面是源码获取入口，不是已固定版本的一键安装器。SAM 代码需要支持 `build_sam3_predictor(..., version='sam3.1')`。MoGe-3 需要其官方 FlexGEMM 等依赖；原运行使用 FlexGEMM `b2fadb29d41846c7981ade6801ffc689fae119cf`，兼容处理见 [`OBJECT_ANNOTATION_RUN.md`](OBJECT_ANNOTATION_RUN.md)。FoundationPose 需要编译其 CUDA 扩展。各模块环境可能不兼容，建议保留独立环境。

## 运行手型估计

```bash
export H2R_ROOT="$(pwd)"
export PYTHONPATH="$H2R_ROOT:$H2R_ROOT/vendor/hamer${PYTHONPATH:+:$PYTHONPATH}"

# 用已有 MediaPipe 环境为原始完整视频生成逐帧 JSON。
# 这是恢复的历史检测入口，会保留全部原始帧于内存；长视频需评估内存。
HUMANEGO_INPUT=/absolute/path/input.mp4 \
HAND_LANDMARKER_MODEL="$H2R_ROOT/models/mediapipe/hand_landmarker.task" \
HUMANEGO_HAND_OUT="$H2R_ROOT/outputs/hands" \
python -m human2robot.v2.run_mediapipe_hand

python -m human2robot.hand_mesh.identity \
  --input outputs/hands/input_hand.json --output outputs/hands/tracked.json

# HaMeR 的配置使用 _DATA 相对路径，因此从此目录启动。
cd "$H2R_ROOT/models/hamer"
python -m human2robot.hand_mesh.infer \
  --video /absolute/path/input.mp4 \
  --hands "$H2R_ROOT/outputs/hands/tracked.json" \
  --checkpoint "$H2R_ROOT/models/hamer/hamer_ckpts/checkpoints/hamer.ckpt" \
  --hamer-root "$H2R_ROOT/vendor/hamer" \
  --out "$H2R_ROOT/outputs/hand_mesh_new"
```

`input_hand.json` 中 `input` 对应输入文件名。完整模型环境、校验与几何边界见 [`HAND_MESH.md`](HAND_MESH.md)。

## 运行视频分割与深度

```bash
export PYTHONPATH="$PWD:$PWD/vendor/sam3-main:$PWD/vendor/MoGe-main${PYTHONPATH:+:$PYTHONPATH}"
python -m human2robot.object_annotation.sam3_video \
  --video /absolute/path/input.mp4 \
  --checkpoint "$PWD/models/sam3.1/sam3.1_multiplex.pt" \
  --output outputs/sam_new --prompt "T-shirt"
python -m human2robot.object_annotation.depth_video \
  --video /absolute/path/input.mp4 \
  --checkpoint "$PWD/models/moge-3-vitl/model.pt" \
  --output outputs/depth_new
```

`sam3_video` 是折衣样本的单目标入口；其他任务应使用自己的提示配置和 `unified_multimodal`。统一批处理支持 `--sam-checkpoint`、`--depth-checkpoint`，或 `SAM3_CHECKPOINT`、`MOGE_CHECKPOINT` 环境变量，默认读取当前仓库的 `models/`。`--stage preflight` 中的“锅至少检测 27/30 帧”仅适用于原来的 `umi_hifi_pot` 已知样本，不能作为任意视频的通用验收。

`tools/run_batch30*.sh` 是历史 30 条样本作业入口。已支持 `H2R_ROOT`、`H2R_OUT`、`H2R_PYTHON`，但仍需要原样本清单、分片与提示 JSON；源码发布不包含这些原始数据。`local_annotation_lab/unified` 中部分构建脚本仍保存实验路径，迁移时按目录中的说明修改。

## 监控与质量边界

推理入口将实际进度写入对应输出目录的 `events/`。例如：

```bash
tensorboard --logdir outputs --host 127.0.0.1 --port 16069
```

打开 http://127.0.0.1:16069/ 。该地址需在运行机器启动 TensorBoard 后才可访问。跨机器使用自己的端口转发。模型文件检查与源码单元测试不启动模型推理，也不产生训练 loss。

所有模型输出仍需质检；文件与校验值齐全只能证明准备完整，不能证明分割、几何或动作语义正确。30 条样本的历史结果仍是初标，包含已确认的局部问题。

## 许可与来源

- 项目代码：MIT。
- MediaPipe：Apache-2.0；保留 Google 上游来源和许可。
- HaMeR：MIT，权重来源记录为 `Leo-TX/hamer`；MANO 是独立资产。
- MoGe-3 权重：官方模型卡标注 MIT；DINOv2 等依赖代码有各自许可。
- SAM 3.1：SAM License；再分发必须附上原许可，不能换成项目的 MIT。
- FoundationPose：NVIDIA Source Code License，限制为非商业研究/评估。
- MANO：使用者通过 https://mano.is.tue.mpg.de/ 获取；此发布不上传 `MANO_RIGHT.pkl`。
- Wan2.1-VACE-1.3B：官方 Apache-2.0 仓库，按官方完整快照下载。

许可证原文保存在 [`model_assets/licenses/`](../model_assets/licenses/)。这些第三方模型均保持原作者归属；未声称是本项目训练的新模型。

## 维护者补传和公开发布

[`tools/publish_model_assets.py`](../tools/publish_model_assets.py) 使用已有的 Hugging Face 登录，不在源码中保存 token。先只读检查目标仓库，确认现有文件的来源与许可，然后在持有上述模型文件的机器执行发布：

```bash
python -m pip install huggingface_hub
# 通过 hf auth login 或当前进程的 HF_TOKEN 配置有写入权限的账户。
python tools/publish_model_assets.py --root /path/to/existing/project \
  --sam-root /path/to/existing/sam3.1 --report outputs/hub_before.json

# 上传缺失/不匹配的清单文件、说明和许可；随后设为公开并验证匿名访问。
python tools/publish_model_assets.py --root /path/to/existing/project \
  --sam-root /path/to/existing/sam3.1 --publish --make-public \
  --report outputs/hub_release.json
```

该工具要求目标仓库已经存在，不自动新建同名仓库。上传前验证所有本地资产，保留旧 README 的归档，并使用远端提交号防止覆盖其他并发更新。MANO 不在上传清单中。已有的其他文件不会被删除；将私有仓库公开前，维护者还需核对那些已有文件。
