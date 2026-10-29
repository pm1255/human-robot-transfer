"""Bridge the shared SAM3 predictor to the SAM3.1 init_state signature.

The upstream base predictor unconditionally forwards offload_state_to_cpu,
although the multiplex model does not accept it. Only omit unsupported false
options; a requested capability must never silently disappear.
"""
import inspect
import time
import uuid
from types import MethodType


def propagate_prompted_video(predictor, session_id, start, frames, seed_result):
    """A last-frame prompt must run backward before upstream considers it done.

    The backward iterator excludes its seed. Emit the actual add_prompt output
    once, then request actual tracker inference for every preceding frame.
    """
    at_end = frames > 1 and start == frames - 1
    if at_end:
        assert seed_result['frame_index'] == start
        yield seed_result
    direction = 'backward' if at_end else 'both' if start else 'forward'
    yield from predictor.handle_stream_request(dict(
        type='propagate_in_video', session_id=session_id,
        start_frame_index=start, propagation_direction=direction,
        evict_cached_frame_outputs=True))


def install_interactive_output_adapter(predictor):
    """Keep real tracker masks when a new frame has no detector cache.

    SAM3.1's interactive path can start from a spatial prompt without running
    text detection first. Upstream _build_sam2_output returns {} for uncached
    frames before merging the newly computed tracker masks. The adapter only
    corrects that merge; it does not manufacture, propagate or fill any mask.
    """
    original = predictor.model._build_sam2_output

    def merge(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        if frame_idx not in inference_state['cached_frame_outputs'] and refined_obj_id_to_mask:
            assert all(mask is not None for mask in refined_obj_id_to_mask.values())
            return dict(refined_obj_id_to_mask)
        return original(inference_state, frame_idx, refined_obj_id_to_mask)

    predictor.model._build_sam2_output = MethodType(merge, predictor.model)


def install_session_adapter(predictor):
    parameters = inspect.signature(predictor.model.init_state).parameters
    if 'offload_state_to_cpu' in parameters:
        return False

    def start_session(self, resource_path, session_id=None,
                      offload_video_to_cpu=False, offload_state_to_cpu=False):
        if offload_state_to_cpu:
            raise ValueError('This SAM3.1 model cannot offload its tracking state')
        options = dict(resource_path=resource_path,
                       offload_video_to_cpu=offload_video_to_cpu)
        for name in ('async_loading_frames', 'video_loader_type'):
            if name in parameters and hasattr(self, name):
                options[name] = getattr(self, name)
        state = self.model.init_state(**options)
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        self._all_inference_states[sid] = dict(
            state=state, session_id=sid, start_time=now, last_use_time=now)
        return {'session_id': sid}

    predictor.start_session = MethodType(start_session, predictor)
    return True
