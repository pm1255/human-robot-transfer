"""CPU z-buffer with smooth vertex lighting and 2x antialiasing."""

import numpy as np
import cv2
from numba import njit


@njit(cache=True)
def _raster(points, depths, colors, width, height, perspective):
    out = np.zeros((height, width, 3), np.uint8)
    zbuf = np.full((height, width), 1e30)
    mask = np.zeros((height, width), np.uint8)
    for k in range(len(points)):
        p = points[k]
        z = depths[k]
        c = colors[k]
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-8:
            continue
        xa = max(0, int(np.floor(min(x0, x1, x2))))
        xb = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        ya = max(0, int(np.floor(min(y0, y1, y2))))
        yb = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 1):
                a = ((y1 - y2) * (x + 0.5 - x2) + (x2 - x1) * (y + 0.5 - y2)) / den
                b = ((y2 - y0) * (x + 0.5 - x2) + (x0 - x2) * (y + 0.5 - y2)) / den
                cc = 1 - a - b
                if a < -0.00001 or b < -0.00001 or cc < -0.00001:
                    continue
                if perspective:
                    zz = 1 / (a / z[0] + b / z[1] + cc / z[2])
                    a = a / z[0] * zz
                    b = b / z[1] * zz
                    cc = cc / z[2] * zz
                else:
                    zz = a * z[0] + b * z[1] + cc * z[2]
                if zz >= zbuf[y, x]:
                    continue
                zbuf[y, x] = zz
                mask[y, x] = 255
                for ch in range(3):
                    out[y, x, ch] = min(
                        255, max(0, int(a * c[0, ch] + b * c[1, ch] + cc * c[2, ch]))
                    )
    return out, mask


def render(points, depths, colors, w, h, perspective=False):
    im, mask = _raster(
        np.asarray(points, float) * 2,
        np.asarray(depths, float),
        np.asarray(colors, float),
        w * 2,
        h * 2,
        perspective,
    )
    # Premultiplied RGB plus coverage; avoid dark seams at triangle boundaries.
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    alpha = mask.astype(float) / 255
    good = alpha > 0
    im[good] = np.clip(im[good].astype(float) / alpha[good, None], 0, 255).astype(
        np.uint8
    )
    return im, mask


def lighting(normals, bgr, light):
    diffuse = np.clip(normals @ light, 0, 1)
    rim = np.abs(normals[..., 2])
    return np.clip(
        (0.44 + 0.56 * diffuse[..., None]) * np.asarray(bgr)
        + (0.12 * rim[..., None] ** 12) * 255,
        0,
        255,
    )
