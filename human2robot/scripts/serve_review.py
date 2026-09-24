"""Loopback-only review server with byte ranges for accurate video seeking."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        path = Path(self.translate_path(self.path))
        if path.is_file() and self.headers.get("Range"):
            size = path.stat().st_size
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
            if not m or not (m[1] or m[2]):
                self.send_error(416)
                return None
            start = int(m[1]) if m[1] else max(0, size - int(m[2]))
            end = min(size - 1, int(m[2])) if m[1] and m[2] else size - 1
            if not 0 <= start <= end < size:
                self.send_error(416)
                return None
            f = path.open("rb")
            f.seek(start)
            self.byte_range = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(self.byte_range))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        return super().send_head()

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_range
        while remaining:
            data = source.read(min(65536, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)

    def log_message(self, *args):
        pass


p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8877)
p.add_argument("--directory", default=str(Path(__file__).resolve().parents[1]))
a = p.parse_args()
from functools import partial

ThreadingHTTPServer(
    ("127.0.0.1", a.port), partial(Handler, directory=a.directory)
).serve_forever()
