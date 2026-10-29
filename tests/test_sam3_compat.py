import unittest
from types import SimpleNamespace
from human2robot.object_annotation.sam3_compat import install_session_adapter, install_interactive_output_adapter, propagate_prompted_video


class SessionCompatibilityTest(unittest.TestCase):
    def test_final_frame_prompt_runs_backward_and_keeps_real_seed(self):
        requests=[]
        seed={'frame_index':2,'outputs':{'mask':'actual_seed'}}
        def stream(request):
            requests.append(request)
            return iter([{'frame_index':1,'outputs':{}},{'frame_index':0,'outputs':{}}])
        result=list(propagate_prompted_video(SimpleNamespace(handle_stream_request=stream),'s',2,3,seed))
        self.assertEqual([r['frame_index']for r in result],[2,1,0])
        self.assertIs(result[0],seed)
        self.assertEqual(requests[0]['propagation_direction'],'backward')

    def test_new_frame_retains_tracker_result_without_detector_cache(self):
        class Model:
            def _build_sam2_output(self, state, frame_idx, refined=None):
                if frame_idx not in state['cached_frame_outputs']:
                    return {}
                return {**state['cached_frame_outputs'][frame_idx], **(refined or {})}
        predictor = SimpleNamespace(model=Model())
        install_interactive_output_adapter(predictor)
        mask = object()
        state = {'cached_frame_outputs': {0: {2: 'previous'}}}
        self.assertEqual(predictor.model._build_sam2_output(state, 1, {1: mask}), {1: mask})
        self.assertEqual(predictor.model._build_sam2_output(state, 0, {1: mask}), {2: 'previous', 1: mask})
        self.assertEqual(predictor.model._build_sam2_output(state, 2), {})
        self.assertNotIn(1, state['cached_frame_outputs'])

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
