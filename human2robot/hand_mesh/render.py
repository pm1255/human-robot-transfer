"""MANO triangle rasterizer in the HaMeR/OpenCV camera convention.

Vertices are already unmirrored by HaMeR inference. Translation is added once;
left-hand face winding is reversed once. There is no convex hull or keypoint mesh.
No OpenGL/CUDA is required for replaying a saved model prediction.
"""
import numpy as np
from numba import njit


def project(vertices, camera_translation, focal, width, height):
    xyz = np.asarray(vertices, dtype=np.float64) + np.asarray(camera_translation)
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 1e-6):
        raise ValueError("Non-finite vertices or mesh behind the camera")
    return np.column_stack((focal * xyz[:, 0] / xyz[:, 2] + width / 2,
                            focal * xyz[:, 1] / xyz[:, 2] + height / 2)), xyz


def vertex_normals(vertices, faces):
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(normals, faces[:, j], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


@njit(cache=True)
def rasterize(uv, xyz, colors, faces, rgb, depth, owner, label):
    height, width = depth.shape
    for face in faces:
        a, b, c = face
        x0, y0 = uv[a]; x1, y1 = uv[b]; x2, y2 = uv[c]
        det = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(det) < 1e-9:
            continue
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(width-1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(height-1, int(np.ceil(max(y0, y1, y2))))
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                px, py = x+0.5, y+0.5
                w0 = ((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/det
                w1 = ((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/det
                w2 = 1-w0-w1
                if min(w0,w1,w2) < -1e-7:
                    continue
                q0, q1, q2 = w0/xyz[a,2], w1/xyz[b,2], w2/xyz[c,2]
                z = 1/(q0+q1+q2)
                if z < depth[y,x]:
                    depth[y,x] = z; owner[y,x] = label
                    for k in range(3):
                        rgb[y,x,k] = (q0*colors[a,k]+q1*colors[b,k]+q2*colors[c,k])*z


def render_hands(hands, faces, focal, width, height):
    rgb = np.zeros((height,width,3), dtype=np.float32)
    depth = np.full((height,width), np.inf, dtype=np.float64)
    owner = np.full((height,width), -1, dtype=np.int32)
    for index, hand in enumerate(hands):
        vertices = np.asarray(hand['vertices'], dtype=np.float64)
        if vertices.shape != (778,3):
            raise ValueError(f"Expected MANO (778,3), received {vertices.shape}")
        if hand['handedness'] not in ('Left','Right'):
            raise ValueError('Unknown handedness')
        f = faces if hand['handedness']=='Right' else faces[:,[0,2,1]]
        uv, xyz = project(vertices, hand['cam_t'], focal, width, height)
        normals = vertex_normals(vertices, f)
        view = -xyz/np.linalg.norm(xyz, axis=1, keepdims=True)
        # Two-sided material makes the open wrist visible without changing topology.
        normals *= np.where(np.sum(normals*view,axis=1,keepdims=True)<0,-1,1)
        light = np.array([-0.35,-0.5,-1.]); light /= np.linalg.norm(light)
        diffuse = np.maximum(0,normals@light)
        half = view+light; half /= np.maximum(np.linalg.norm(half,axis=1,keepdims=True),1e-9)
        spec = np.maximum(0,np.sum(normals*half,axis=1))**36
        color = np.array([0.32,0.70,0.96]) if hand['handedness']=='Left' else np.array([0.78,0.65,0.95])
        colors = np.clip(color*(0.38+0.60*diffuse[:,None])+0.20*spec[:,None],0,1)
        rasterize(uv,xyz,colors,f.astype(np.int64),rgb,depth,owner,index)
    return rgb, depth, owner


def composite(frame_bgr, rgb, depth, opacity=0.92):
    result=frame_bgr.copy(); mask=np.isfinite(depth)
    result[mask]=np.clip((1-opacity)*result[mask]+opacity*255*rgb[mask,::-1],0,255).astype(np.uint8)
    return result
