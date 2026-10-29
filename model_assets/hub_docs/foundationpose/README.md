# FoundationPose refiner and scorer

Upstream: https://github.com/NVlabs/FoundationPose . Original weight download: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i . Copyright NVIDIA Corporation and affiliates.

Both `2023-10-28-18-33-37` (refiner) and `2024-01-11-20-02-45` (scorer) contain `model_best.pth` and `config.yml`. These files use the [NVIDIA Source Code License](../licenses/FoundationPose.txt), limited to non-commercial research/evaluation; they are not covered by the companion project's MIT license.

Companion GitHub script: [foundation_cloth.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/object_annotation/foundation_cloth.py). [Setup and usage](https://github.com/pm1255/human-robot-transfer/blob/main/docs/MODEL_RELEASE.md). Cloth output is a rigid global proxy, not deformable-object ground truth.
