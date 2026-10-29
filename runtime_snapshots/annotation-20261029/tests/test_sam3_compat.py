import unittest
from types import SimpleNamespace
from human2robot.object_annotation.sam3_compat import install_session_adapter


class SessionCompatibilityTest(unittest.TestCase):
    def test_multiplex_session_preserves_supported_options(self):
        class Model:
            def init_state(self, resource_path, offload_video_to_cpu=False,
                           async_loading_frames=False):
                return dict(path=resource_path, cpu=offload_video_to_cpu,
                            asynchronous=async_loading_frames)
        predictor = SimpleNamespace(model=Model(), async_loading_frames=True,
                                    _all_inference_states={})
        self.assertTrue(install_session_adapter(predictor))
        response = predictor.start_session('/frames', session_id='test', offload_video_to_cpu=True)
        self.assertEqual(response, {'session_id': 'test'})
        self.assertEqual(predictor._all_inference_states['test']['state'],
                         dict(path='/frames', cpu=True, asynchronous=True))
        with self.assertRaises(ValueError):
            predictor.start_session('/frames', offload_state_to_cpu=True)

    def test_native_supported_session_is_unchanged(self):
        class Model:
            def init_state(self, resource_path, offload_state_to_cpu=False):
                pass
        sentinel = object()
        predictor = SimpleNamespace(model=Model(), start_session=sentinel)
        self.assertFalse(install_session_adapter(predictor))
        self.assertIs(predictor.start_session, sentinel)


if __name__ == '__main__':
    unittest.main()
