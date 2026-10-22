# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container loopback to a task-owned Unix socket; no arbitrary host forwarding."""

import argparse
import http.client
import http.server
import json
import os
import socket
from pathlib import Path


class UnixConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


def serve(
    *, socket_path: str, port_file: Path, route: str = "/v1/chat/completions", max_body_bytes: int = 16 * 1024 * 1024
) -> None:
    """Expose only the creator chat route within this container's loopback."""

    if route not in {"/v1/chat/completions", "/v1/responses"}:
        raise ValueError("Only explicit model protocol routes are supported")
    if not 1 <= max_body_bytes <= 32 * 1024 * 1024:
        raise ValueError("Invalid relay body limit")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            if self.path != route:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > max_body_bytes:
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
        temporary = port_file.with_name(port_file.name + ".ready")
        temporary.write_text(json.dumps({"port": server.server_port, "pid": os.getpid()}))
        temporary.replace(port_file)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--route", default="/v1/chat/completions")
    parser.add_argument("--max-body-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    serve(socket_path=args.socket, port_file=args.port_file, route=args.route, max_body_bytes=args.max_body_bytes)
