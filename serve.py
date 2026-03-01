"""
serve.py — No-cache HTTP server for the output/ directory.

Starts on port 8765. Silently exits if the port is already in use
(a previous server instance is still running and serving).
All responses include Cache-Control: no-store so the browser always
fetches fresh content after each main.py run.

Usage:
    python serve.py        # starts server and opens route_map.html
    python serve.py --bg   # starts in background, no browser open
"""
import http.server
import os
import sys
import webbrowser
import threading
import socket
import socketserver

PORT = 8765
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with no-cache headers on every response."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress per-request noise


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    open_browser = "--bg" not in sys.argv

    if _port_in_use(PORT):
        # Another instance is already serving — just open the browser if needed
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{PORT}/route_map.html")
        return  # exit cleanly; old server keeps running

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
            url = f"http://127.0.0.1:{PORT}/route_map.html"
            if open_browser:
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
            print(f"Serving output/ at {url}  (Ctrl-C to stop)")
            httpd.serve_forever()
    except OSError:
        # Port was grabbed between check and bind — another instance won the race
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{PORT}/route_map.html")


if __name__ == "__main__":
    main()
