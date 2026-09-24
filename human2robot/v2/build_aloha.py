"""RoboTwin Aloha asset export and fresh IK. Never reuse demo-arm joints."""

import json, hashlib, copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from h2r.kinematics import SerialRobot, vector
from h2r.geometry import transform

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "data/aloha"
OUT = ROOT / "web/v2data/aloha"


def origin(e):
    return (
        transform(
            Rotation.from_euler("xyz", vector(e.get("rpy"), [0, 0, 0])).as_matrix(),
            vector(e.get("xyz"), [0, 0, 0]),
        )
        if e is not None
        else np.eye(4)
    )


class Aloha:
    def __init__(self):
        self.tree = ET.parse(ASSET / "model.urdf").getroot()
        self.robot = SerialRobot(ASSET / "arm.urdf", "fl_base_link", "tcp")
        self.joints = [
            j
            for j in self.tree.findall("joint")
            if j.get("name") in [f"fl_joint{i}" for i in range(1, 9)]
        ]

    def links(self, q, opening):
        result = {"fl_base_link": np.eye(4)}
        for j in self.joints:
            n = int(j.get("name").replace("fl_joint", ""))
            axis = vector(j.find("axis").get("xyz"), [1, 0, 0])
            motion = (
                transform(Rotation.from_rotvec(axis * q[n - 1]).as_matrix())
                if n <= 6
                else transform(translation=axis * opening)
            )
            result[j.find("child").get("link")] = (
                result[j.find("parent").get("link")] @ origin(j.find("origin")) @ motion
            )
        return result


def assets():
    OUT.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ASSET / "model.urdf").getroot()
    arm = ET.Element("robot", name="RoboTwin_Aloha_AgileX_left_arm")
    links = ["fl_base_link"] + [f"fl_link{i}" for i in range(1, 9)]
    for l in tree.findall("link"):
        if l.get("name") in links:
            arm.append(copy.deepcopy(l))
    for j in tree.findall("joint"):
        if j.get("name") in [f"fl_joint{i}" for i in range(1, 9)]:
            arm.append(copy.deepcopy(j))
    ET.SubElement(arm, "link", name="tcp")
    j = ET.SubElement(arm, "joint", name="tcp_fixed", type="fixed")
    ET.SubElement(j, "parent", link="fl_link6")
    ET.SubElement(j, "child", link="tcp")
    ET.SubElement(j, "origin", xyz="0.12 0 0", rpy="0 0 0")
    # This derived URDF supports visualization/FK only; source stays intact.
    for link in arm.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
    for mesh in arm.findall(".//mesh"):
        mesh.set("filename", Path(mesh.get("filename")).name)
    ET.ElementTree(arm).write(ASSET / "arm.urdf")
    groups = []
    for l in arm.findall("link"):
        for vis in l.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            name = Path(m.get("filename")).name
            scene = trimesh.load(ASSET / name, force="scene")
            mesh = scene.to_geometry()
            # Surface simplification preserves concavities, unlike former convex hulls.
            if len(mesh.faces) > 3500:
                mesh = mesh.simplify_quadric_decimation(face_count=3500)
            mesh.apply_transform(origin(vis.find("origin")))
            groups.append(
                {
                    "link": l.get("name"),
                    "vertices": np.round(mesh.vertices, 7).tolist(),
                    "faces": mesh.faces.tolist(),
                    "source": name,
                    "sha256": hashlib.sha256((ASSET / name).read_bytes()).hexdigest(),
                }
            )
    (OUT / "meshes.json").write_text(
        json.dumps(
            {
                "robot": "RoboTwin Aloha / AgileX ARX5 follower arm",
                "source": str(ASSET.resolve()),
                "urdf_sha256": hashlib.sha256(
                    (ASSET / "model.urdf").read_bytes()
                ).hexdigest(),
                "geometry": "original visual surface meshes, decimated to <=3500 triangles per link; not convex hulls",
                "groups": groups,
            },
            separators=(",", ":"),
        )
    )


def main():
    assets()
    a = Aloha()
    index = json.loads((ROOT / "web/data/index.json").read_text())
    manifest = []
    seed = np.array([0, 0.6, 1.3, 0, -0.6, 0])
    home = a.robot.fk(seed)
    for item in index["items"]:
        if not item.get("data"):
            continue
        d = json.loads((ROOT / "web" / item["data"]).read_text())
        poses = np.array(d["repaired_pose_world"], float)
        good = [i for i, p in enumerate(poses) if np.isfinite(p).all()]
        if not good:
            continue
        alignment = home @ np.linalg.inv(poses[good[0]])
        rows = []
        prev = seed.copy()
        for i, p in enumerate(poses):
            if not np.isfinite(p).all():
                rows.append(None)
                continue
            target = alignment @ p
            sol = a.robot.ik(target, seed=prev, previous=prev, dt=0.1, max_nfev=70)
            prev = sol["q"]
            gap = d["aperture_m"][i]
            # Joint translation is an estimated jaw mapping, not calibrated contact width.
            opening = float(np.clip((gap or 0) / 2, 0, 0.04765))
            rows.append(
                {
                    "q": prev.tolist(),
                    "gripper_joint_m": opening,
                    "estimated_human_gap_m": gap,
                    "links": {k: v.tolist() for k, v in a.links(prev, opening).items()},
                    "target": target.tolist(),
                    "actual": sol["actual"].tolist(),
                    "error_m": sol["position_error_m"],
                    "rotation_error_rad": sol["rotation_error_rad"],
                    "ik_valid": sol["valid"],
                    "training_eligible": False,
                }
            )
        result = {
            "id": item["id"],
            "robot": "RoboTwin Aloha / AgileX ARX5 left follower arm",
            "joint_names": a.robot.names,
            "joint_limits_source": "URDF simulation limits ±10 rad and 1000 rad/s; not hardware safe limits",
            "alignment": alignment.tolist(),
            "timestamp": d["timestamp"],
            "frames": rows,
            "training_eligible": False,
            "limitations": [
                "first-frame illustrative alignment, no camera/metric calibration",
                "gripper gap mapped to slide travel without contact calibration",
                "no self/scene collision, balance or hardware validation",
            ],
        }
        (OUT / (item["id"] + ".json")).write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False)
        )
        manifest.append({**item, "aloha": f'v2data/aloha/{item["id"]}.json'})
        print(item["id"], sum(r is not None for r in rows), flush=True)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
