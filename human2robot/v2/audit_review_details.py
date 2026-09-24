"""Recompute rendered FK and development metrics from frozen artifacts."""

import json
from pathlib import Path
import numpy as np
from h2r.kinematics import SerialRobot

root = Path(__file__).resolve().parents[1]
robot = SerialRobot(root / "configs/demo_arm.urdf")
index = json.loads((root / "web/data/index.json").read_text())
frames = 0
cases = 0
for item in index["items"]:
    if not item.get("data"):
        continue
    data = json.loads((root / "web" / item["data"]).read_text())
    cases += 1
    assert not any(data["robot_training_valid"])
    for i, q in enumerate(data["robot_joint_reference"]):
        if not np.isfinite(np.asarray(q, float)).all():
            continue
        pose, points = robot.fk(q, True)
        np.testing.assert_allclose(pose, data["robot_actual_pose"][i], atol=1e-10)
        np.testing.assert_allclose(points, data["robot_link_points"][i], atol=1e-10)
        target = np.asarray(data["target_pose_robot"][i], float)
        error = np.linalg.norm(pose[:3, 3] - target[:3, 3])
        np.testing.assert_allclose(error, data["ik_position_error_m"][i], atol=1e-10)
        frames += 1
long = json.loads((root / "web/data/long_gap.json").read_text())
assert not any(long["human_valid"][30:40])
assert not any(long["geometry_valid"][30:40])
assert all(
    not np.isfinite(np.asarray(q, float)).any()
    for q in long["robot_joint_reference"][30:40]
)
rows = json.loads((root / "web/v2data/development_rollouts.json").read_text())
j = [i for i in range(14) if i not in [6, 13]]
for r in rows:
    assert r["diagnostic"] and r["seed"] == 100001
    assert (root / "web" / r["video"]).is_file()
    d = json.loads((root / "web" / r["trace"]).read_text())
    a = np.array(d["action"])
    s = np.array(d["state"])
    assert len(a) == r["steps"]
    np.testing.assert_allclose(
        np.abs(a[:, j] - s[:, j]).mean(), r["command_state_joint_mae_rad"], rtol=1e-5
    )
    if "physical_after" in d:
        np.testing.assert_allclose(
            np.abs(a[:, j] - np.array(d["physical_after"])[:, j]).mean(),
            r["command_physical_after_joint_mae_rad"],
            rtol=1e-5,
        )
    np.testing.assert_allclose(
        np.abs(np.diff(a[:, j], n=3, axis=0)).mean(),
        r["joint_third_difference_per_command"],
        rtol=1e-5,
    )
result = {
    "passed": True,
    "retarget_cases": cases,
    "recomputed_fk_frames": frames,
    "development_rollouts": len(rows),
    "checks": [
        "displayed chain and endpoint match FK",
        "IK translation residual matches displayed metric",
        "long gaps remain invalid and contain no robot pose",
        "development action metrics recomputed",
        "development videos excluded from formal benchmark",
    ],
    "not_tested": [
        "human 3D accuracy",
        "grasp/contact feasibility",
        "real robot execution",
        "formal closed-loop success",
    ],
}
(root / "outputs/v2/review_details_audit.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(json.dumps(result, ensure_ascii=False))
