from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class Progress:
    def __init__(self, root, stage, total):
        from torch.utils.tensorboard import SummaryWriter
        self.root, self.stage, self.total = Path(root), stage, total
        self.writer = SummaryWriter(str(self.root / 'events' / stage), flush_secs=5)
        self.started = time.time()

    def update(self, done, status='running', **details):
        record = dict(stage=self.stage, status=status, completed=done, total=self.total,
                      updated_at_unix=time.time(), elapsed_seconds=time.time()-self.started, **details)
        atomic_json(self.root / f'{self.stage}_status.json', record)
        self.writer.add_scalar('progress/processed_frames', done, done)
        self.writer.add_scalar('progress/completion', done / max(1, self.total), done)
        self.writer.add_scalar('state/running', float(status == 'running'), done)
        self.writer.add_scalar('state/waiting', float(status.startswith('waiting')), done)
        self.writer.add_scalar('state/failed', float(status == 'failed'), done)
        self.writer.add_text('status', json.dumps(record, ensure_ascii=False), done)
        for key, value in details.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, done)
        self.writer.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def close(self):
        self.writer.close()
