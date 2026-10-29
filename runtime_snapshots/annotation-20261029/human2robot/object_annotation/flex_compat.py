"""Expose the dtype byte-width property used by pinned FlexGEMM on Triton 3.3.

Calculate the Python integer outside a JIT expression: Triton 3.3 cannot
consistently combine its constexpr wrapper with raw dtype attribute arithmetic.
No kernels or numerical operations are changed.
"""
import triton
import triton.language as tl


@triton.jit
def _bitwise_or(a,b):
    return a | b


@triton.jit
def _reduce_or(value,axis:tl.constexpr=0):
    return tl.reduce(value,axis,_bitwise_or)


def install_dtype_itemsize():
    if not hasattr(tl.dtype, 'itemsize'):
        tl.dtype.itemsize = property(lambda dtype: tl.constexpr(dtype.primitive_bitwidth // 8))
    for dtype, expected in ((tl.uint8, 1), (tl.int16, 2), (tl.int32, 4), (tl.int64, 8)):
        if dtype.itemsize != expected:
            raise RuntimeError(f'Unexpected Triton dtype byte width for {dtype}')
    if not hasattr(tl,'reduce_or'):
        tl.reduce_or=_reduce_or
