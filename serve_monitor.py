#!/usr/bin/env python3
"""Serve the repo directory so the live monitor can read outputs/ over HTTP.

Run this from the clean_version/ directory, then open:
    http://localhost:8000/dashboard/monitor.html
The dashboard polls /outputs/programbench_auction_subrun_harness_agents/ every ~4s.
"""
import http.server, socketserver, os, sys
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Live monitor: http://localhost:{PORT}/dashboard/monitor.html")
    httpd.serve_forever()
