import unittest
import numpy as np
from human2robot.object_annotation.cloth_geometry import backproject, reference_surface, pose_quality


class ClothGeometryTest(unittest.TestCase):
    def setUp(self):
        self.depth=np.ones((48,64),np.float32)
        self.K=np.array([[100.,0,31.5],[0,100.,23.5],[0,0,1]])
        self.mask=np.ones(self.depth.shape,bool)

    def test_projection_and_partial_surface(self):
        points=backproject(self.depth,self.K)
        projected=points@self.K.T;projected=projected[...,:2]/projected[...,2:]
        v,u=np.indices(self.depth.shape)
        np.testing.assert_allclose(projected,np.stack((u,v),axis=-1),atol=1e-5)
        self.mask[:,28:36]=False
        vertices,faces,centre,used=reference_surface(self.depth,self.K,self.mask,stride=2)
        # No triangle bridges the excluded cloth gap.
        cols=used[faces]%(64//2)*2
        self.assertTrue(np.all(np.max(cols,axis=1)-np.min(cols,axis=1)<=2))
        self.assertTrue(np.isfinite(vertices).all())

    def test_bad_alignment_never_accepted(self):
        pose=np.eye(4)
        self.assertTrue(pose_quality(pose,self.depth,self.depth,self.mask)['accepted'])
        self.assertFalse(pose_quality(pose,np.zeros_like(self.depth),self.depth,self.mask)['accepted'])
        self.assertFalse(pose_quality(pose,self.depth*2,self.depth,self.mask)['accepted'])
        bad=pose.copy();bad[0,0]=-1
        self.assertFalse(pose_quality(bad,self.depth,self.depth,self.mask)['accepted'])
        jump=pose.copy();jump[:2,:2]=[[0,-1],[1,0]]
        self.assertFalse(pose_quality(jump,self.depth,self.depth,self.mask,pose)['accepted'])


if __name__=='__main__':unittest.main()
