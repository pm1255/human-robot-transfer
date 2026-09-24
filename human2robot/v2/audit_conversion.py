"""Validate exported media and G1 FK against saved fitting targets, not ground truth."""

import json
import hashlib
import cv2
import mujoco
import numpy as np
from v2.render_conversion import ROOT, G1


def main():
    rows = json.loads((ROOT / "web/v2data/conversion/manifest.json").read_text())
    robot = G1()
    counts = {"videos": 0, "frames": 0, "robot_frames": 0, "g1_frames": 0}
    errors = []
    rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    np.testing.assert_allclose(np.linalg.det(rotation), 1)
    for row in rows:
        data = json.loads((ROOT / "web" / row["audit"]).read_text())
        assert data["training_eligible"] is False and data["trained_steps"] == 0
        from pathlib import Path

        assert (
            hashlib.sha256(Path(data["source"]).read_bytes()).hexdigest()
            == data["source_sha256"]
        )
        capture = cv2.VideoCapture(str(ROOT / "web" / row["video"]))
        assert capture.isOpened()
        assert (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            == len(data["frames"])
            == row["frames"]
        )
        assert capture.get(cv2.CAP_PROP_FPS) == 10
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape[1] % 3 == 0
            decoded += 1
        capture.release()
        assert decoded == row["frames"]
        counts["videos"] += 1
        for i, frame in enumerate(data["frames"]):
            counts["frames"] += 1
            assert frame["frame"] == i and frame["timestamp"] == i / 10
            assert frame["training_eligible"] is False
            assert frame["unmodified_pixels_verified"]
            if not frame["has_robot"]:
                assert frame["render_pixels"] == 0
                continue
            counts["robot_frames"] += 1
            q = np.asarray(frame["qpos"])
            assert np.isfinite(q).all()
            if row["kind"] != "body":
                aloha = json.loads(
                    (ROOT / "web/v2data/aloha" / (row["id"] + ".json")).read_text()
                )["frames"][i]
                np.testing.assert_allclose(q, aloha["q"], atol=1e-10)
                np.testing.assert_allclose(
                    frame["ik_error_m"], aloha["error_m"], atol=1e-10
                )
                assert frame["robot"] == "RoboTwin Aloha / AgileX ARX5"
                continue
            counts["g1_frames"] += 1
            assert q.shape == (36,)
            assert (q[7:] >= robot.ranges[:, 0] - 1e-6).all()
            assert (q[7:] <= robot.ranges[:, 1] + 1e-6).all()
            np.testing.assert_allclose(np.linalg.norm(q[3:7]), 1)
            robot.d.qpos[:] = q
            mujoco.mj_forward(robot.m, robot.d)
            good = np.array(frame["good"])
            residual = np.linalg.norm(
                (robot.d.xpos[robot.bids] - np.array(frame["target"]))[good], axis=1
            ).mean()
            np.testing.assert_allclose(residual, frame["fit_error_m"], atol=1e-10)
            errors.append(float(residual))
    result = {
        "passed": True,
        **counts,
        "g1_fit_mean_m": float(np.mean(errors)),
        "checks": [
            "all video frames decode",
            "source SHA256 matches",
            "timestamps and 10 fps match",
            "G1 joint limits and unit root quaternion",
            "G1 fitting error independently recomputed from FK",
            "hand replacement uses the new Aloha joints and errors",
            "unmodified pixels verified before labels and lossy encoding",
            "no preview accepted for training",
        ],
        "not_validated": [
            "human pose accuracy",
            "mask and object preservation accuracy",
            "camera and metric scale",
            "depth occlusion and contact",
            "collision, balance and dynamic feasibility",
            "robot task success",
            "VLA benefit",
        ],
    }
    (ROOT / "outputs/v2/conversion_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
