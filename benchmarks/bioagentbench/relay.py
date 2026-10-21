# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container loopback to a task-owned Unix socket; no arbitrary host forwarding."""

import argparse
import http.client
import http.server
import json
import socket
from pathlib import Path


class UnixConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


def serve(*, socket_path: str, port_file: Path) -> None:
    """Expose only the creator chat route within this container's loopback."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16 * 1024 * 1024:
                self.send_error(413)
                return
            upstream = UnixConnection(socket_path, timeout=600)
            try:
                upstream.request(
                    "POST",
                    self.path,
                    self.rfile.read(length),
                    {"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")},
                )
                response = upstream.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                upstream.close()

        def log_message(self, format, *args):
            pass  # Native prompts, headers and local bearer tokens are not logged.

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        port_file.write_text(json.dumps({"port": server.server_port}))
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--port-file", type=Path, required=True)
    args = parser.parse_args()
    serve(socket_path=args.socket, port_file=args.port_file)
