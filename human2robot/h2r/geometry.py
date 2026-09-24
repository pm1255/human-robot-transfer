"""Right handed, metres, active rotations; T_A_B maps B coordinates into A."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(rotation=None, translation=None):
    out = np.eye(4)
    if rotation is not None:
        out[:3, :3] = rotation
    if translation is not None:
        out[:3, 3] = translation
    return out


def check_transform(T, atol=1e-5):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Transform must be finite 4x4")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("Invalid homogeneous row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol) or abs(np.linalg.det(R)-1) > atol:
        raise ValueError("Rotation must be SO(3); reflections need explicit mirror handling")
    return T


def invert(T):
    T = check_transform(T)
    R = T[:3, :3]
    return transform(R.T, -R.T @ T[:3, 3])


def apply(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def unit(v, epsilon=1e-7):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < epsilon:
        raise ValueError("Degenerate direction")
    return v / norm


def pinch_frame(joints):
    """Heuristic pinch candidate, NOT a contact or force estimator.

    x: thumb-tip -> index-tip (jaw closing line), z: wrist -> MCP plane,
    orthogonalized against x. y=z cross x. Thumb/index tips: 4/8; MCP: 5/17.
    """
    j = np.asarray(joints, float)
    if j.shape != (21, 3) or not np.isfinite(j[[0, 4, 8, 5, 17]]).all():
        raise ValueError("Need finite wrist, thumb/index tips and two MCP landmarks")
    x = unit(j[8]-j[4])
    along = (j[5]+j[17])/2-j[0]
    z = unit(along-x*np.dot(x, along))
    y = unit(np.cross(z, x))
    R = np.column_stack([x, y, z])
    return transform(R, (j[4]+j[8])/2), float(np.linalg.norm(j[8]-j[4]))


def transfer_object_goal(T_source_object, T_source_grasp, T_target_object, T_grasp_tcp=None):
    """Same corresponding object frame/geometry; grasp adaptation precedes this.

    T_grasp_tcp is the fixed transform from robot TCP coordinates to the
    chosen grasp reference. This operation alone does not select a valid grasp.
    """
    offset = np.eye(4) if T_grasp_tcp is None else check_transform(T_grasp_tcp)
    return check_transform(T_target_object) @ invert(T_source_object) @ check_transform(T_source_grasp) @ offset


def interp_rotations(t, rotations, query):
    return Slerp(t, Rotation.from_matrix(rotations))(query).as_matrix()


def rotation_error(target, actual):
    return Rotation.from_matrix(target @ actual.T).as_rotvec()
