"""Credential-free, exact-model proxy for host solver and Docker verifier.

Bind only loopback and an explicitly supplied Docker bridge address. Upstream
is the existing credential-free worker tunnel, never a public provider URL.
"""
import argparse
import ipaddress
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def validate(path, body):
    if not isinstance(body, dict):
        raise ValueError("Expected an object")
    if path not in ("/v1/responses", "/v1/chat/completions"):
        raise ValueError("Unsupported API path")
    if body.get("model") != MODEL or body.get("stream"):
        raise ValueError("Only exact-model nonstream requests are allowed")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"transport-only"}')

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("Invalid request length")
            raw = self.rfile.read(length)
            body = json.loads(raw)
            validate(self.path, body)
        except (ValueError, TypeError):
            self.send_error(400, "Request rejected")
            return
        request = urllib.request.Request(
            "http://127.0.0.1:18883" + self.path, data=raw,
            headers={"Content-Type": "application/json",
                     "X-Campaign-Benchmark": "NatureBench"})
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        except (urllib.error.URLError, OSError):
            # No retry: the upstream may already have processed this request.
            self.send_error(502, "Uncertain upstream transport failure; inspect receipts")
            return
        if status == 200:
            try:
                if json.loads(data).get("model") != MODEL:
                    raise ValueError("Returned model mismatch")
            except (ValueError, AttributeError):
                self.send_error(502, "Invalid upstream model response; inspect receipts")
                return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-bridge", required=True)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.docker_bridge)
    if not address.is_private or address.is_unspecified or address.is_loopback:
        raise ValueError("Provide the actual private Docker bridge address")
    servers = [ThreadingHTTPServer((host, 18885), Handler)
               for host in ("127.0.0.1", str(address))]
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
