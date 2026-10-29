"""Adapt type annotations and equivalent syntax for pinned FlexGEMM/Triton 3.3.

Triton 3.3 interprets the class tl.pointer_type as an invalid concrete argument
signature ('ier_type'). tl.tensor lets it infer the actual pointer from the
argument, as done by the package's other kernels. Also parenthesize boolean
chains and pass reshape dimensions as tuples. Casts remain untouched.
"""
import argparse
import ast
import re
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
args=parser.parse_args()
for path in sorted((args.source/'flex_gemm').rglob('*.py')):
    source=path.read_text()
    updated,count=re.subn(r'(:\s*)tl\.pointer_type(?:\s*\|\s*None)?(?=\s*[,)=])',r'\1tl.tensor',source)
    updated,unions=re.subn(r'(:\s*)tl\.tensor\s*\|\s*None(?=\s*[,)=])',r'\1tl.tensor',updated)
    count+=unions
    old='vec.dtype.itemsize == 1 or vec.dtype.itemsize == 2 or vec.dtype.itemsize == 4'
    if old in updated:
        updated=updated.replace(old,'(vec.dtype.itemsize == 1 or vec.dtype.itemsize == 2) or vec.dtype.itemsize == 4')
        count+=1
    for old,new in [
        ('tl.reshape(query_vec_32, *query_vec_32.shape[:-1])','tl.reshape(query_vec_32, query_vec_32.shape[:-1])'),
        ('tl.reshape(tl.cast(vec, tl.uint16, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 2, 2)',
         'tl.reshape(tl.cast(vec, tl.uint16, bitcast=True), (*vec.shape[:-1], vec.shape[-1] // 2, 2))'),
        ('tl.reshape(tl.cast(vec, tl.uint8, bitcast=True), *vec.shape[:-1], vec.shape[-1] // 4, 4)',
         'tl.reshape(tl.cast(vec, tl.uint8, bitcast=True), (*vec.shape[:-1], vec.shape[-1] // 4, 4))'),
    ]:
        if old in updated:
            updated=updated.replace(old,new)
            count+=1
    for divisor in [2,4]:
        old=f'(*vec.shape[:-1], vec.shape[-1] // {divisor}, {divisor})'
        new=f'vec.shape[:-1] + [vec.shape[-1] // {divisor}, {divisor}]'
        if old in updated:
            updated=updated.replace(old,new);count+=1
    if count:
        ast.parse(updated)
        backup=path.with_suffix('.py.triton33-original')
        if not backup.exists():backup.write_text(source)
        path.write_text(updated)
        print(f'{path.relative_to(args.source)}: {count} compatibility substitutions')
