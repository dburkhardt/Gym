"""Campaign-only judge gateway: exact model allowlist and full request/response receipts."""
import json
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'
GATEWAY = 'http://127.0.0.1:49180/v1'


class JudgeOutputBudgetExhausted(ValueError):
    """A completed judge HTTP response exhausted its generation budget."""


def validate_result(result):
    if result.get('model') != MODEL:
        raise ValueError('Unexpected returned judge model')
    choices = result.get('choices', [])
    if choices and all(x.get('finish_reason') in ('stop', 'tool_calls', 'length') for x in choices) and any(x.get('finish_reason') == 'length' for x in choices):
        raise JudgeOutputBudgetExhausted('Judge output token budget exhausted')
    if not choices or any(x.get('finish_reason') not in ('stop', 'tool_calls') for x in choices):
        raise ValueError('Judge incomplete or malformed')
    if not all(((x.get('message', {}).get('content') or '').strip() or x.get('message', {}).get('tool_calls')) for x in choices):
        raise ValueError('Judge empty')


@contextmanager
def proxy(*, key_file: Path, ledger: Path, model: str, base_url: str):
    if model != MODEL or base_url.rstrip('/') != GATEWAY:
        raise ValueError('Campaign requires exact Super model and local gateway')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    key = key_file.read_text().strip()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            receipt = {}
            try:
                if self.path != '/v1/chat/completions': raise ValueError('Unsupported route')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4000000: raise ValueError('Request size outside campaign bound')
                payload = json.loads(self.rfile.read(size))
                if payload.get('model') != MODEL or payload.get('stream'): raise ValueError('Unexpected model or streaming')
                receipt['request'] = payload
                request = urllib.request.Request(GATEWAY + '/chat/completions', data=json.dumps(payload).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=600) as response: result = json.load(response)
                receipt['response'] = result
                validate_result(result)
                body, status = json.dumps(result).encode(), 200
            except Exception as exc:
                receipt['failure'] = str(exc)
                body, status = json.dumps({'error':{'message':str(exc)}}).encode(), 400
            with ledger.open('a') as out: out.write(json.dumps(receipt)+'\n')
            self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/v1'
    finally: server.shutdown();server.server_close();thread.join()
