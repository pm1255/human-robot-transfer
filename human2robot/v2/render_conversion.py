"""Auditable RGB-to-robot visual preview. No learned inpainting, no dynamics claim.
CPU rasterizer uses actual robot visual surfaces and their own FK.
All composites are rejected for training: uncalibrated camera/masks/contact.
"""

import argparse, json, subprocess, hashlib
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
import trimesh
from v2.surface_renderer import render as raster, lighting
from scipy.spatial.transform import Rotation
import mujoco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/v2data/conversion"
XML = ROOT.parent / "work/Light-O1/examples/sonic/assets/g1/g1_29dof.xml"
BODY_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]


def label(im, text, y=23):
    cv2.rectangle(im, (0, y - 22), (im.shape[1], y + 10), (29, 43, 35), -1)
    cv2.putText(
        im,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 245, 233),
        1,
        cv2.LINE_AA,
    )


def valid(p):
    return np.isfinite(np.asarray(p, float)).all()


def xy(points, w, h):
    return np.rint(np.asarray(points)[:, :2] * [w, h]).astype(np.int32)


def mask_body(frame, w, h):
    b = np.asarray(frame["body"], float)
    mask = np.zeros((h, w), np.uint8)
    if b.shape != (33, 4):
        return mask
    ps = xy(b, w, h)
    good = (
        (b[:, 2] > 0.6)
        & (b[:, 3] > 0.6)
        & (b[:, 0] >= 0)
        & (b[:, 0] <= 1)
        & (b[:, 1] >= 0)
        & (b[:, 1] <= 1)
    )
    width = max(9, int(np.linalg.norm(ps[11] - ps[12]) * 0.45))
    for a, c in BODY_EDGES:
        if good[a] and good[c]:
            cv2.line(mask, tuple(ps[a]), tuple(ps[c]), 255, width, cv2.LINE_AA)
    if good[[11, 12, 23, 24]].all():
        cv2.fillConvexPoly(mask, cv2.convexHull(ps[[11, 12, 23, 24]]), 255)
    if good[0]:
        cv2.circle(
            mask,
            tuple(ps[0] - [0, int(width * 0.35)]),
            max(14, int(width * 1.0)),
            255,
            -1,
        )
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))


class G1:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.map = {
            11: "left_shoulder_pitch_link",
            12: "right_shoulder_pitch_link",
            13: "left_elbow_link",
            14: "right_elbow_link",
            15: "left_wrist_yaw_link",
            16: "right_wrist_yaw_link",
            23: "left_hip_pitch_link",
            24: "right_hip_pitch_link",
            25: "left_knee_link",
            26: "right_knee_link",
            27: "left_ankle_roll_link",
            28: "right_ankle_roll_link",
        }
        self.ids = list(self.map)
        self.bids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.map[i])
            for i in self.ids
        ]
        self.mesh = []
        for i in range(self.m.ngeom):
            if (
                self.m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH
                or self.m.geom_group[i] == 1
            ):
                continue
            mid = self.m.geom_dataid[i]
            start = self.m.mesh_vertadr[mid]
            n = self.m.mesh_vertnum[mid]
            verts = self.m.mesh_vert[start : start + n].astype(float)
            fa = self.m.mesh_faceadr[mid]
            nf = self.m.mesh_facenum[mid]
            faces = self.m.mesh_face[fa : fa + nf].copy()
            mesh = trimesh.Trimesh(verts, faces, process=False)
            if len(faces) > 2200:
                mesh = mesh.simplify_quadric_decimation(face_count=2200)
            self.mesh.append(
                (
                    i,
                    np.asarray(mesh.vertices)[np.asarray(mesh.faces)],
                    np.asarray(mesh.vertex_normals)[np.asarray(mesh.faces)],
                )
            )
        self.prev = np.zeros(32)
        self.ranges = self.m.jnt_range[1:]
        self.d.qpos[:3] = 0
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[7:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.neutral = self.d.xpos[self.bids].copy()

    def setq(self, q):
        self.d.qpos[:3] = 0
        quat = Rotation.from_rotvec(q[:3]).as_quat()
        self.d.qpos[3:7] = quat[[3, 0, 1, 2]]
        self.d.qpos[7:] = q[3:]
        mujoco.mj_forward(self.m, self.d)

    def fit(self, f):
        body = np.asarray(f["body_local_xyz"], float)
        b = np.asarray(f["body"], float)
        if body.shape != (33, 3) or b.shape != (33, 4):
            return None
        good = (b[self.ids, 2] > 0.6) & (b[self.ids, 3] > 0.6)
        if good.sum() < 8:
            return None
        center = (body[23] + body[24]) / 2
        v = body - center
        human = np.c_[-v[:, 2], v[:, 0], -v[:, 1]]
        # Proper rotation: det=+1. Retarget directions, using ROBOT segment lengths.
        target = np.zeros((33, 3))
        neutral = {j: self.neutral[k] for k, j in enumerate(self.ids)}
        unit = lambda x: x / max(1e-8, np.linalg.norm(x))
        hc = (neutral[23] + neutral[24]) / 2
        sc = (neutral[11] + neutral[12]) / 2
        hd = unit(human[23] - human[24])
        sd = unit(human[11] - human[12])
        torso = unit((human[11] + human[12]) / 2 - (human[23] + human[24]) / 2)
        target[23] = hc + hd * np.linalg.norm(neutral[23] - neutral[24]) / 2
        target[24] = hc - hd * np.linalg.norm(neutral[23] - neutral[24]) / 2
        target[11] = (
            hc
            + torso * np.linalg.norm(sc - hc)
            + sd * np.linalg.norm(neutral[11] - neutral[12]) / 2
        )
        target[12] = (
            hc
            + torso * np.linalg.norm(sc - hc)
            - sd * np.linalg.norm(neutral[11] - neutral[12]) / 2
        )
        for src, dst in [
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ]:
            target[dst] = target[src] + unit(human[dst] - human[src]) * np.linalg.norm(
                neutral[dst] - neutral[src]
            )
        target = target[self.ids]
        prev = self.prev.copy()

        def residual(q):
            self.setq(q)
            return np.r_[
                ((self.d.xpos[self.bids] - target)[good] * 8).ravel(),
                (q - prev) * 0.06,
                q[3:] * 0.012,
            ]

        lo = np.r_[[-3.14] * 3, self.ranges[:, 0]]
        hi = np.r_[[3.14] * 3, self.ranges[:, 1]]
        result = least_squares(
            residual,
            np.clip(prev, lo + 1e-6, hi - 1e-6),
            bounds=(lo, hi),
            max_nfev=22,
            ftol=1e-4,
        )
        self.prev = result.x
        self.setq(result.x)
        # Weak-perspective least squares in the source image, not a calibrated camera.
        model = self.d.xpos[self.bids]
        projected = np.c_[model[:, 1], -model[:, 2]]
        pts = b[self.ids, :2]
        P = projected[good]
        Y = pts[good]
        scale = np.sum((P - P.mean(0)) * (Y - Y.mean(0))) / max(
            1e-9, np.sum((P - P.mean(0)) ** 2)
        )
        # Pixel isotropy will be fitted using actual image size in render.
        return {
            "qpos": self.d.qpos.copy().tolist(),
            "target": target.tolist(),
            "good": good.tolist(),
            "fit_error_m": float(
                np.linalg.norm(model[good] - target[good], axis=1).mean()
            ),
            "model_points": model.copy(),
            "image_points": pts.copy(),
        }

    def render(self, rec, w, h):
        model = rec["model_points"]
        good = np.array(rec["good"])
        P = np.c_[model[:, 1], -model[:, 2]][good]
        Y = rec["image_points"][good] * [w, h]
        s = np.sum((P - P.mean(0)) * (Y - Y.mean(0))) / max(
            1e-9, np.sum((P - P.mean(0)) ** 2)
        )
        origin = Y.mean(0) - s * P.mean(0)
        triangles = []
        cols = []
        light = np.array([0.8, -0.3, 1.0])
        light /= np.linalg.norm(light)
        for gid, local, normals in self.mesh:
            rotation = self.d.geom_xmat[gid].reshape(3, 3)
            tri = local @ rotation.T + self.d.geom_xpos[gid]
            normals = normals @ rotation.T
            bgr = (self.m.geom_rgba[gid, :3] * 215 + 25)[::-1]
            cols.extend(lighting(normals, bgr, light))
            triangles.extend(tri)
        tri = np.asarray(triangles)
        ps = np.stack([tri[:, :, 1], -tri[:, :, 2]], -1) * s + origin
        image, mask = raster(ps, 5 - tri[:, :, 0], np.asarray(cols), w, h)
        repro = np.c_[model[:, 1], -model[:, 2]] * s + origin
        rec["projection_error_px"] = float(
            np.linalg.norm(repro[good] - Y, axis=1).mean()
        )
        rec["weak_camera"] = {"pixel_scale": float(s), "origin_xy": origin.tolist()}
        return image, mask


_aloha_meshes = None


def hand_preview(d, i, w, h):
    global _aloha_meshes
    kp = d.get("keypoints", [])[i]
    pts = kp.get("landmarks_2d") if kp else None
    mask = np.zeros((h, w), np.uint8)
    image = np.zeros((h, w, 3), np.uint8)
    rm = mask.copy()
    rec = d["_aloha"]["frames"][i]
    if pts is None or rec is None:
        return mask, image, rm, None
    ps = xy(pts, w, h)
    cv2.fillConvexPoly(mask, cv2.convexHull(ps), 255)
    wrist = ps[0]
    direction = wrist - ps[9]
    direction = direction / max(1, np.linalg.norm(direction))
    width = max(12, int(np.linalg.norm(ps[5] - ps[17]) * 1.1))
    end = wrist + direction * max(w, h)
    cv2.line(mask, tuple(wrist), tuple(end.astype(int)), 255, width, cv2.LINE_AA)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    if _aloha_meshes is None:
        groups = json.loads((ROOT / "web/v2data/aloha/meshes.json").read_text())[
            "groups"
        ]
        _aloha_meshes = []
        for g in groups:
            m = trimesh.Trimesh(g["vertices"], g["faces"], process=False)
            _aloha_meshes.append(
                (g["link"], m.vertices[m.faces], m.vertex_normals[m.faces])
            )
    inv = np.linalg.inv(np.asarray(d["_aloha"]["alignment"]))
    K = np.asarray(d["metadata"]["intrinsics"])
    sw, sh = d["metadata"]["image_size"]
    tris = []
    colors = []
    light = np.array([-0.3, -0.5, -1.0])
    light /= np.linalg.norm(light)
    for link, local, normals in _aloha_meshes:
        T = inv @ np.asarray(rec["links"][link])
        tri = local @ T[:3, :3].T + T[:3, 3]
        good = (tri[:, :, 2] > 0.04).all(1)
        tri = tri[good]
        n = (normals @ T[:3, :3].T)[good]
        col = (
            np.array([185, 185, 180])
            if link.endswith(("7", "8"))
            else np.array([103, 81, 65])
        )
        colors.extend(lighting(n, col, light))
        tris.extend(tri)
    tri = np.asarray(tris)
    colors = np.asarray(colors)
    if len(tri):
        project = tri @ K.T
        ps = project[:, :, :2] / project[:, :, 2, None] * [w / sw, h / sh]
        image, rm = raster(ps, tri[:, :, 2], colors, w, h, True)
    return (
        mask,
        image,
        rm,
        {
            "qpos": rec["q"],
            "ik_error_m": rec["error_m"],
            "geometry_valid": rec["ik_valid"],
            "robot": "RoboTwin Aloha / AgileX ARX5",
            "gripper_joint_m": rec["gripper_joint_m"],
        },
    )


def write_case(kind, name, video, annotation, engine=None, max_frames=90):
    cap = cv2.VideoCapture(str(video))
    records = []
    frames = []
    poster = None
    for i in range(
        min(
            max_frames,
            len(annotation["frames"] if kind == "body" else annotation["timestamp"]),
        )
    ):
        ok, raw = cap.read()
        if not ok:
            break
        h, w = raw.shape[:2]
        if kind == "body":
            f = annotation["frames"][i]
            mask = mask_body(f, w, h)
            rec = engine.fit(f)
            if rec:
                render, rm = engine.render(rec, w, h)
            else:
                render = np.zeros_like(raw)
                rm = np.zeros((h, w), np.uint8)
                mask[:] = 0
        else:
            mask, render, rm, rec = hand_preview(annotation, i, w, h)
        # No action inferred on missing frames, and no output accepted as training data.
        cleaned = cv2.inpaint(raw, mask, 3, cv2.INPAINT_TELEA) if rec else raw.copy()
        replaced = cleaned.copy()
        alpha = rm.astype(float) / 255.0
        replaced = (
            replaced * (1 - alpha[..., None]) + render * alpha[..., None]
        ).astype(np.uint8)
        marked = raw.copy()
        marked[mask > 0] = (
            marked[mask > 0] * 0.35 + np.array([40, 50, 240]) * 0.65
        ).astype(np.uint8)
        panels = [raw.copy(), marked, replaced.copy()]
        for panel, title in zip(
            panels,
            [
                "1 SOURCE RGB",
                "2 DELETION MASK (HEURISTIC)",
                "3 ROBOT REPLACEMENT (PREVIEW)",
            ],
        ):
            label(panel, title)
        for panel in panels:
            label(panel, f"{name}  t={i/10:.1f}s  NOT TRAINING DATA", h - 12)
        row = np.concatenate(panels, axis=1)
        frames.append(row)
        r = {
            "frame": i,
            "timestamp": i / 10,
            "has_robot": rec is not None,
            "training_eligible": False,
            "mask_pixels": int((mask > 0).sum()),
            "render_pixels": int((rm > 0).sum()),
            "unmodified_pixels_verified": bool(
                np.array_equal(
                    raw[(mask == 0) & (rm == 0)], replaced[(mask == 0) & (rm == 0)]
                )
            ),
        }
        if rec:
            r.update(
                {
                    k: v
                    for k, v in rec.items()
                    if k not in ["model_points", "image_points"]
                }
            )
        records.append(r)
        if poster is None and rec:
            poster = row.copy()
    cap.release()
    if not frames:
        raise RuntimeError(name + " no decoded frames")
    target = OUT / (name + ".mp4")
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        "10",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    cv2.imwrite(str(OUT / (name + ".jpg")), poster if poster is not None else frames[0])
    (OUT / (name + ".json")).write_text(
        json.dumps(
            {
                "kind": kind,
                "source": str(video),
                "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "method": (
                    "bounded G1 IK + original visual mesh surface rendering"
                    if kind == "body"
                    else "new Aloha URDF IK + original visual mesh surface rendering"
                ),
                "background": "OpenCV Telea, single-frame, heuristic mask",
                "trained_steps": 0,
                "training_eligible": False,
                "limitations": [
                    "uncalibrated camera and scale",
                    "heuristic foreground mask may erase objects or leave human pixels",
                    "no scene depth or occlusion/contact/dynamics validation",
                    "background temporally inconsistent; no VLA trained on composites",
                ],
                "frames": records,
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(name, len(frames), sum(r["has_robot"] for r in records), flush=True)
    return {
        "id": name,
        "kind": kind,
        "frames": len(frames),
        "robot_frames": sum(r["has_robot"] for r in records),
        "video": f"v2data/conversion/{name}.mp4",
        "poster": f"v2data/conversion/{name}.jpg",
        "audit": f"v2data/conversion/{name}.json",
        "training_eligible": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    hands = (
        ["epic_000004"]
        if a.smoke
        else [
            "epic_000004",
            "epic_000007",
            "epic_000015",
            "epic_000040",
            "epic_000100",
            "epic_000250",
            "epic_000500",
            "epic_001000",
        ]
    )
    for name in hands:
        d = json.loads((ROOT / "web/data" / f"{name}.json").read_text())
        d["_aloha"] = json.loads(
            (ROOT / "web/v2data/aloha" / f"{name}.json").read_text()
        )
        items.append(
            write_case(
                "hand",
                name,
                ROOT / "web/media" / f"{name}.mp4",
                d,
                max_frames=8 if a.smoke else 90,
            )
        )
    bodies = (
        ["hmdb_body_golf_03"]
        if a.smoke
        else [
            "hmdb_body_golf_03",
            "hmdb_body_golf_02",
            "hmdb_body_kick_ball_00",
            "hmdb_body_cartwheel_00",
        ]
    )
    for name in bodies:
        d = json.loads((ROOT / "web/v2data/annotations" / f"{name}.json").read_text())
        items.append(
            write_case(
                "body",
                name,
                ROOT / "web/v2data/media" / f"{name}.mp4",
                d,
                G1(),
                max_frames=8 if a.smoke else 90,
            )
        )
    (OUT / "manifest.json").write_text(json.dumps(items, indent=2))
    print("published", len(items))


if __name__ == "__main__":
    main()
