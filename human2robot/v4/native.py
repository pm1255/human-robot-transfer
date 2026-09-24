"""Import pinned official modules without importing optional Wan application dependencies."""
import sys, types, importlib
import torch
from v4.common import VENDOR

def module(name):
    if 'h2r_wan' not in sys.modules:
        pkg = types.ModuleType('h2r_wan'); pkg.__path__ = [str(VENDOR / 'wan/modules')]
        sys.modules['h2r_wan'] = pkg
    m = importlib.import_module('h2r_wan.' + name)
    if name == 'model': m.flash_attention = sdpa
    return m

def sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
         q_scale=None, causal=False, window_size=(-1, -1), **kwargs):
    assert window_size == (-1, -1) and q_lens is None
    dtype = v.dtype; q = q.to(dtype); k = k.to(dtype)
    if q_scale is not None: q = q * q_scale
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1], device=k.device)[None, :] < k_lens.to(k.device)[:, None]
        mask = mask[:, None, None, :]
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout_p, is_causal=causal,
        scale=softmax_scale).transpose(1, 2).contiguous()
