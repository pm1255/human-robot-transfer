"""Explicit states for stages waiting on another stage or a failed preflight."""
import argparse
from pathlib import Path
from .common import Progress

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--stage',required=True)
parser.add_argument('--status',required=True)
parser.add_argument('--reason',required=True)
parser.add_argument('--total',type=int,default=4834)
args=parser.parse_args()
progress=Progress(args.output,args.stage,args.total)
progress.update(0,args.status,reason=args.reason)
progress.close()
