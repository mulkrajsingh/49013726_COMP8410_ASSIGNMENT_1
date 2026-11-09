# Serve the local saved-output review page and accept its JSON save requests.
# Read the 108-entry manifest; write a draft or completed review after validation.
# This helper stores student entries; it does not create judgments or attest for them.

from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit
from tempfile import TemporaryDirectory
import json,os,threading
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'additional_analysis/student_review'
test_directory = TemporaryDirectory(prefix='comp8410_review_test_')
TEST = Path(test_directory.name)
manifest=json.loads((OUT/'review_manifest.json').read_text());expected={(r['job_id'],r['criterion']) for r in manifest};LOCK=threading.Lock()
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
 def do_GET(self):
  rel=unquote(urlsplit(self.path).path).lstrip('/');p=(ROOT/rel).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or p.suffix not in {'.html','.png','.mp4','.glb'} or not rel.startswith(('additional_analysis/student_review/','data/prepared/','outputs/')):self.send_error(404);return
  return super().do_GET()
 def respond(self,code,d):
  b=json.dumps(d).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_POST(self):
  if self.path not in ['/api/review/save','/api/test/save'] or self.headers.get('Host')!='127.0.0.1:8732' or self.headers.get('Origin') not in [None,'http://127.0.0.1:8732']:self.respond(403,{'error':'Local review access only'});return
  try:
   length=int(self.headers.get('Content-Length',0))
   if not 0<length<500000:raise ValueError('Invalid request size')
   d=json.loads(self.rfile.read(length));records=d['records'];actual=[(r['job_id'],r['criterion']) for r in records]
   if d.get('review_version')!='additional_student_visual_v1' or len(actual)!=108 or set(actual)!=expected:raise ValueError('Review does not match the 108 supplied criteria')
   if any(r.get('judgment') not in ['', 'pass','issue','unassessable'] for r in records):raise ValueError('Invalid judgment')
   # Completion requires the student's attestation and all required fields.
   complete=d.get('completed') is True
   if complete and (d.get('student_attested') is not True or any(not all(r.get(k) for k in ['reviewer','reviewed_at','judgment','observation']) for r in records)):raise ValueError('Completed review is incomplete')
   folder=TEST if self.path=='/api/test/save' else OUT
   filename='student_visual_review_completed.json' if complete else 'student_visual_review_draft.json';p=folder/filename
   with LOCK:
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
   self.respond(200,{'filename':filename,'path':str(p),'completed':complete})
  except (ValueError,KeyError,TypeError) as e:self.respond(400,{'error':str(e)})
 def log_message(self,*args):pass
print('Review page server: http://127.0.0.1:8732/additional_analysis/student_review/REVIEW_SAVED_OUTPUTS.html',flush=True)
try:
 with ThreadingHTTPServer(('127.0.0.1',8732),Handler) as server:
  server.serve_forever()
finally:
 test_directory.cleanup()
