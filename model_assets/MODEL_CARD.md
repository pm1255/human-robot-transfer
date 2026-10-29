---
license: other
license_name: component-specific-licenses
license_link: https://github.com/pm1255/human-robot-transfer/blob/main/docs/MODEL_RELEASE.md
language:
- en
- zh
tags:
- robotics
- hand-pose-estimation
- image-segmentation
- depth-estimation
- model-collection
---

# Embodied Data Loop — Model Assets

**配套代码 / Companion code: [pm1255/human-robot-transfer](https://github.com/pm1255/human-robot-transfer).**

This repository collects third-party pretrained assets used by the Human ↔ Robot
Transfer Lab annotation pipelines. These are upstream models, not a newly
trained end-to-end model. Each component retains its authorship and license.
The current file list and `manifest.json` describe which files have actually
been included; a model collection is not loaded through one Transformers call.

## 从模型找到 GitHub 脚本

| 模型 / Component | 配套 GitHub 脚本 | 功能 |
|---|---|---|
| MediaPipe | [tools/run.py](https://github.com/pm1255/human-robot-transfer/blob/main/tools/run.py), [run_mediapipe_hand.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/v2/run_mediapipe_hand.py) | 手/人体关键点与基础检测 |
| HaMeR | [hand_mesh/infer.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/infer.py), [identity.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/identity.py), [render.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/render.py) | 三维手型、身份修复与表面渲染 |
| SAM 3.1 | [sam3_video.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/sam3_video.py), [unified_multimodal.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/unified_multimodal.py) | 视频分割与掩码派生框 |
| MoGe-3 ViT-L | [depth_video.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/depth_video.py) | 单目深度 |
| FoundationPose | [foundation_cloth.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/foundation_cloth.py) | 物体位姿 / 衣物刚体代理试验 |
| 质量检查（无独立权重） | [confidence.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/confidence.py), [batch30_validate.py](https://github.com/pm1255/human-robot-transfer/blob/main/local_annotation_lab/unified/batch30_validate.py) | 一致性筛查与完整性检查 |
| 统一标注网页 | [local_annotation_lab](https://github.com/pm1255/human-robot-transfer/tree/main/local_annotation_lab), [serve_hand_review.py](https://github.com/pm1255/human-robot-transfer/blob/main/tools/serve_hand_review.py) | 查看原视频及各类标注 |

## Download and verify

```bash
git clone https://github.com/pm1255/human-robot-transfer.git
cd human-robot-transfer
python -m pip install huggingface_hub
python tools/model_assets.py download --component mediapipe --component hamer --component sam31 --component moge3 --component foundationpose
python tools/model_assets.py check
```

Add `--revision <Hub commit>` to pin this repository version. The download tool
uses the SHA-256 manifest and installs weights/configuration files into the
paths expected by the GitHub scripts. This validates file integrity, not model
accuracy. See the [complete setup and usage guide](https://github.com/pm1255/human-robot-transfer/blob/main/docs/MODEL_RELEASE.md)
for environment requirements and executable examples.

## Coverage and required external assets

- The core asset inventory is approximately 7.97 GB, excluding environments,
  raw datasets, MANO, optional VACE, and robot meshes.
- HaMeR needs its checkpoint, adjacent `model_config.yaml`, mean parameters,
  and **MANO_RIGHT.pkl obtained separately by the user** from
  [the MANO website](https://mano.is.tue.mpg.de/). MANO is not redistributed here.
- FoundationPose requires both networks and their `config.yml` files.
- Optional robot-to-human video editing uses the complete official
  [Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)
  snapshot. [setup_vace.sh](https://github.com/pm1255/human-robot-transfer/blob/main/scripts/setup_vace.sh)
  obtains it from the original repository.
- Aloha/G1 retargeting additionally requires robot geometry assets described
  in [QUICKSTART.md](https://github.com/pm1255/human-robot-transfer/blob/main/docs/QUICKSTART.md).
- This project does **not** provide newly trained general semantic segmentation
  of action timelines, physical contact, affordance, or grasp-prediction weights.
  Existing timeline labels and 2D contact cues are sample-specific annotations
  and rules, not ground truth or a separate trained model.

## Licenses and attribution

There is no single MIT license for all weights in this collection. Original
license texts accompany the collection in `licenses/`.

| Component | Original source | License |
|---|---|---|
| MediaPipe | [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 |
| HaMeR | [geopavlakos/hamer](https://github.com/geopavlakos/hamer), checkpoint provenance [Leo-TX/hamer](https://huggingface.co/Leo-TX/hamer) | MIT; MANO separate |
| MoGe-3 | [Ruicheng/moge-3-vitl](https://huggingface.co/Ruicheng/moge-3-vitl) | MIT model card; dependencies retain their own licenses |
| SAM 3.1 | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) | SAM License |
| FoundationPose | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | NVIDIA Source Code License, non-commercial research/evaluation only |

The project's own scripts are MIT-licensed. Predictions can contain missed
detections, incorrect identity/segmentation, uncalibrated monocular geometry,
and uncertain semantic boundaries. Quality scores are screening evidence, not
calibrated probabilities or automatic training approval.
