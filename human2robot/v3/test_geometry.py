import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from v3.common import local_targets,apply_target,category

class GeometryTest(unittest.TestCase):
    def test_local_delta_roundtrip_under_rotated_reference(self):
        rng=np.random.default_rng(331)
        for _ in range(100):
            p=np.r_[rng.normal(size=3),Rotation.random(random_state=rng).as_quat(),rng.random()]
            q=np.r_[rng.normal(size=3),Rotation.random(random_state=rng).as_quat(),rng.random()]
            a=local_targets(p,q[None])[0];got=apply_target(p,a)
            np.testing.assert_allclose(got[:3],q[:3],atol=1e-6)
            self.assertLess((Rotation.from_quat(got[3:7]).inv()*Rotation.from_quat(q[3:7])).magnitude(),1e-6)
            self.assertAlmostEqual(float(got[7]),float(q[7]),places=6)
    def test_rigid_frame_change_does_not_change_local_action(self):
        rng=np.random.default_rng(337)
        for _ in range(40):
            p=np.r_[rng.normal(size=3),Rotation.random(random_state=rng).as_quat(),.1]
            q=np.r_[rng.normal(size=3),Rotation.random(random_state=rng).as_quat(),.7]
            r=Rotation.random(random_state=rng);t=rng.normal(size=3)
            def change(x):return np.r_[r.apply(x[:3])+t,(r*Rotation.from_quat(x[3:7])).as_quat(),x[7]]
            np.testing.assert_allclose(local_targets(p,q[None]),local_targets(change(p),change(q)[None]),atol=1e-6)
    def test_translation_fit_recovers_known_camera_geometry(self):
        from v3.prepare_human import infer_pose
        world=np.zeros((21,3),np.float64)
        world[0]=[0,-.04,0];world[1]=[-.02,-.02,.004];world[2]=[-.03,0,.006]
        world[5]=[.04,.02,0];world[9]=[0,.04,0];world[13]=[-.02,.03,.002];world[17]=[-.04,.02,0]
        world[4]=[-.015,.03,.008];world[8]=[.02,.055,.004]
        world=Rotation.from_euler('xyz',[.2,-.3,.1]).apply(world)
        translation=np.array([.1,.05,.6]);camera=world+translation
        xy=camera[:,:2]/camera[:,2:3]*[1,640/360]+.5
        pose,error=infer_pose(xy,world,(360,640))
        np.testing.assert_allclose(pose[:3],camera[[4,8]].mean(0),atol=1e-6)
        self.assertLess(error,1e-8)
        camera=world+np.array([.1,.05,-.6]);xy=camera[:,:2]/camera[:,2:3]*[1,640/360]+.5
        self.assertIsNone(infer_pose(xy,world,(360,640)))
    def test_task_filter_checks_manipulated_object(self):
        for t in ['Put sausages in the frying pan.','Pick up the lid from the pot.','Wash the cup.','Open the bottle cap.','Pour water into a mug.','Pick up a plastic bag.']:
            self.assertIsNone(category(t),t)
        for t in ['Pick up the cup from the drawer.','Place the mug on the table.','Take the bottle out of the fridge.']:
            self.assertEqual(category(t),'container_transfer',t)
if __name__=='__main__':unittest.main()
