import numpy as np
from v2.surface_renderer import render


def test_near_surface_occludes_far_independent_of_draw_order():
    points = np.array([[[2, 2], [14, 2], [2, 14]]] * 2, float)
    depths = np.array([[2, 2, 2], [1, 1, 1]], float)
    colors = np.array([[[255, 0, 0]] * 3, [[0, 255, 0]] * 3], float)
    a, mask = render(points, depths, colors, 16, 16, True)
    b, _ = render(
        points[::-1].copy(), depths[::-1].copy(), colors[::-1].copy(), 16, 16, True
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[4, 4], [0, 255, 0])
    assert mask[4, 4] == 255 and mask[15, 15] == 0


def test_vertex_lighting_interpolates_interior():
    p = np.array([[[0, 0], [12, 0], [0, 12]]], float)
    z = np.ones((1, 3))
    c = np.array([[[0, 0, 0], [240, 240, 240], [0, 0, 0]]], float)
    im, _ = render(p, z, c, 16, 16)
    assert 70 < im[2, 4, 0] < 110
