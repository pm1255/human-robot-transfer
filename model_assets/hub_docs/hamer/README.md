# HaMeR hand mesh checkpoint

Upstream: https://github.com/geopavlakos/hamer . Checkpoint source: https://huggingface.co/Leo-TX/hamer . These original files retain the upstream MIT license, reproduced in [HaMeR.txt](../licenses/HaMeR.txt).

Companion GitHub scripts: [infer.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/infer.py), [identity.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/identity.py), [render.py](https://github.com/pm1255/human-robot-transfer/blob/main/human2robot/hand_mesh/render.py).

The Hub keeps `hamer.ckpt`, `model_config.yaml`, and `mano_mean_params.npz` in this folder. The [download tool](https://github.com/pm1255/human-robot-transfer/blob/main/tools/model_assets.py) maps them to the nested installation paths expected by HaMeR. `MANO_RIGHT.pkl` must be obtained separately from https://mano.is.tue.mpg.de/ and is not distributed here.

[Setup and usage](https://github.com/pm1255/human-robot-transfer/blob/main/docs/MODEL_RELEASE.md). Hand meshes are predictions, not calibrated 3D truth.
