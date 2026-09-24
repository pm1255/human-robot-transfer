"""Independently check saved Aloha poses, mesh origins and new joint references."""

import json, hashlib
import numpy as np
from h2r.kinematics import SerialRobot
from h2r.geometry import rotation_error
from v2.build_aloha import ROOT, ASSET, OUT


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    count = 0
    changed = 0
    model = json.loads((OUT / "meshes.json").read_text())
    assert len(model["groups"]) == 9
    for g in model["groups"]:
        assert (
            hashlib.sha256((ASSET / g["source"]).read_bytes()).hexdigest()
            == g["sha256"]
        )
        vertices = np.array(g["vertices"])
        faces = np.array(g["faces"])
        assert np.isfinite(vertices).all() and faces.max() < len(vertices)
    for item in manifest:
        d = json.loads((ROOT / "web" / item["aloha"]).read_text())
        old = json.loads((ROOT / "web" / item["data"]).read_text())
        np.testing.assert_allclose(d["timestamp"], old["timestamp"])
        for i, r in enumerate(d["frames"]):
            if r is None:
                continue
            q = np.array(r["q"])
            count += 1
            if not np.allclose(q, old["robot_joint_reference"][i]):
                changed += 1
            for name, T in r["links"].items():
                if name == "fl_base_link":
                    expected = np.eye(4)
                else:
                    robot = SerialRobot(ASSET / "arm.urdf", "fl_base_link", name)
                    values = (
                        q[: robot.n] if robot.n <= 6 else np.r_[q, r["gripper_joint_m"]]
                    )
                    expected = robot.fk(values)
                np.testing.assert_allclose(T, expected, atol=1e-10)
            robot = SerialRobot(ASSET / "arm.urdf", "fl_base_link", "tcp")
            assert (q >= robot.lower).all() and (q <= robot.upper).all()
            actual = robot.fk(q)
            target = np.array(r["target"])
            np.testing.assert_allclose(actual, r["actual"], atol=1e-10)
            np.testing.assert_allclose(
                np.linalg.norm(actual[:3, 3] - target[:3, 3]), r["error_m"], atol=1e-10
            )
            np.testing.assert_allclose(
                np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])),
                r["rotation_error_rad"],
                atol=1e-10,
            )
            assert r["training_eligible"] is False
    assert changed == count
    result = {
        "passed": True,
        "cases": len(manifest),
        "verified_frames": count,
        "fresh_aloha_joint_frames": changed,
        "visual_links": len(model["groups"]),
        "checks": [
            "source mesh SHA256",
            "all link matrices recomputed with serial-chain FK",
            "TCP and errors recomputed",
            "new joint values differ from demo arm",
            "timestamps preserve source alignment",
            "URDF simulation ranges; not hardware validation",
        ],
        "training_eligible": False,
    }
    (ROOT / "outputs/v2/aloha_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
