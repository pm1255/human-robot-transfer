import unittest
import numpy as np
from human2robot.hand_mesh.render import project, rasterize
from human2robot.hand_mesh.quality import assess
from human2robot.hand_mesh.identity import repair


class RenderRegressionTests(unittest.TestCase):
    def test_camera_projection_and_left_reflection(self):
        vertices=np.array([[.1,.2,0],[-.1,-.2,0]])
        uv,_=project(vertices,[0,0,2],100,640,480)
        np.testing.assert_allclose(uv,[[325,250],[315,230]])
        mirrored=vertices.copy();mirrored[:,0]*=-1
        uv_left,_=project(mirrored,[0,0,2],100,640,480)
        np.testing.assert_allclose(uv_left[:,0],640-uv[:,0])
        with self.assertRaises(ValueError):project(vertices,[0,0,-1],100,640,480)

    def test_triangles_preserve_gap_and_use_nearest_surface(self):
        # Two disconnected surfaces: the gap must remain empty, unlike a convex hull.
        uv=np.array([[1.,1.],[4.,1.],[1.,5.],[7.,1.],[10.,1.],[10.,5.]])
        xyz=np.column_stack((uv,np.ones(6)*2))
        rgb=np.zeros((12,12,3),np.float32);depth=np.full((12,12),np.inf);owner=np.full((12,12),-1,np.int32)
        faces=np.array([[0,1,2],[3,4,5]])
        rasterize(uv,xyz,np.tile([1.,0.,0.],(6,1)),faces,rgb,depth,owner,0)
        self.assertTrue(np.isinf(depth[2,5]));self.assertEqual(owner[1,1],0)
        near=xyz.copy();near[:,2]=1
        rasterize(uv,near,np.tile([0.,1.,0.],(6,1)),faces,rgb,depth,owner,1)
        rasterize(uv,xyz,np.tile([1.,0.,0.],(6,1)),faces,rgb,depth,owner,2)
        self.assertEqual(owner[1,1],1);np.testing.assert_allclose(rgb[1,1],[0,1,0])

    def test_topology_is_not_quality_pass(self):
        vertices=np.zeros((778,3));joints=np.zeros((21,3))
        refs=[dict(x=.1,y=.1) for _ in range(21)]
        q=assess(vertices,[0,0,1],joints,refs,100,640,480)
        self.assertEqual(q['status'],'REVIEW')
        self.assertFalse(q['metric_depth_verified'])

    def test_identity_follows_tracks_across_detection_order_and_crossing(self):
        def hand(x,label):return dict(handedness=dict(category=label,score=.95),landmarks=[dict(x=x,y=.5)]*21)
        data={'frames':[]}
        for i in range(21):
            hands=[hand(.2+i*.03,'Left'),hand(.8-i*.03,'Right' if i!=10 else 'Left')]
            if i%2:hands=hands[::-1]
            data['frames'].append(dict(frame_index=i,hands=hands))
        fixed=repair(data)
        for i,f in enumerate(fixed['frames']):
            self.assertEqual(len({h['handedness']['category'] for h in f['hands']}),2)
        self.assertEqual(data['frames'][10]['hands'][1]['handedness']['category'],'Left')
        self.assertEqual(fixed['frames'][10]['hands'][1]['handedness']['category'],'Right')


if __name__=='__main__':unittest.main()
