"""Local review server with byte ranges, required for reliable MP4 seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        path=self.translate_path(self.path)
        if not self.headers.get('Range'):
            self.range_remaining=None
            return super().send_head()
        try:
            f=open(path,'rb');size=f.seek(0,2)
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
            if not match:raise ValueError()
            a,b=match.groups()
            if not a:
                start=max(0,size-int(b));end=size-1
            else:
                start=int(a);end=min(int(b) if b else size-1,size-1)
            if start>end or start>=size:raise ValueError()
        except (OSError,ValueError):
            if 'f' in locals():f.close()
            self.send_error(416,'Invalid byte range');return None
        self.send_response(206);self.send_header('Content-type',self.guess_type(path))
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1));self.end_headers()
        f.seek(start);self.range_remaining=end-start+1;return f
    def copyfile(self,source,outputfile):
        if getattr(self,'range_remaining',None) is None:
            try:return super().copyfile(source,outputfile)
            except (BrokenPipeError,ConnectionResetError):return
        try:
            while self.range_remaining:
                chunk=source.read(min(self.range_remaining,1024*1024))
                if not chunk:break
                outputfile.write(chunk);self.range_remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--port',type=int,default=0)
    a=p.parse_args();server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=a.directory))
    print(f'http://127.0.0.1:{server.server_port}/',flush=True);server.serve_forever()
