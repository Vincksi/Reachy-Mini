"""Reachy Mini - live camera feed app.

Streams the robot's camera (read locally from the running daemon via the
Python SDK) to your browser as an MJPEG stream, with a snapshot button.

Run:
    .\\reachy_mini_env\\Scripts\\activate.ps1
    python camera_app.py

Then open http://127.0.0.1:8001 (it opens automatically).

Requires the Reachy Mini daemon to be running, e.g.:
    reachy-mini-daemon            (real robot)
    reachy-mini-daemon --sim      (MuJoCo simulation - note: sim has no camera)
"""

from __future__ import annotations

import io
import json
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from reachy_mini import ReachyMini

# --- Config ---------------------------------------------------------------
# "127.0.0.1" = localhost only. Set to "0.0.0.0" to view from your phone on the
# same Wi-Fi (this exposes the camera feed on your network with NO authentication).
HOST = "127.0.0.1"
PORT = 8001  # the daemon owns 8000; use a different port here.
JPEG_QUALITY = 80
TARGET_FPS = 30
STALL_TIMEOUT = 4.0  # seconds without a frame before rebuilding the camera reader
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# --- Shared state ---------------------------------------------------------
_latest_jpeg: bytes | None = None
_latest_lock = threading.Lock()
_stop = threading.Event()
_frame_count = 0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Reachy Mini - Camera</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e14; color:#e6e6e6;
         font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         display:flex; flex-direction:column; align-items:center; min-height:100vh; }
  header { padding:16px; font-size:1.15rem; font-weight:600; letter-spacing:.3px; }
  .wrap { width:100%; max-width:960px; padding:0 12px 24px; box-sizing:border-box; }
  .frame { position:relative; width:100%; background:#000; border-radius:12px;
           overflow:hidden; box-shadow:0 8px 30px rgba(0,0,0,.5); }
  img#feed { display:block; width:100%; height:auto; }
  .bar { display:flex; gap:12px; align-items:center; justify-content:center;
         margin-top:16px; flex-wrap:wrap; }
  button { background:#3b82f6; color:#fff; border:none; padding:12px 20px;
           border-radius:10px; font-size:1rem; cursor:pointer; min-height:44px; }
  button:hover { background:#2563eb; }
  .status { font-size:.85rem; opacity:.75; min-height:1.2em; text-align:center; margin-top:12px; }
</style>
</head>
<body>
  <header>Reachy Mini - Live Camera</header>
  <div class="wrap">
    <div class="frame"><img id="feed" src="/stream" alt="camera feed"></div>
    <div class="bar"><button id="snap">Snapshot</button></div>
    <div class="status" id="status"></div>
  </div>
<script>
  const status = document.getElementById('status');
  document.getElementById('snap').addEventListener('click', async () => {
    status.textContent = 'Saving snapshot...';
    try {
      const r = await fetch('/snapshot', { method: 'POST' });
      const j = await r.json();
      status.textContent = j.ok
        ? 'Saved: ' + j.file + '  (' + j.width + ' x ' + j.height + ')'
        : 'Snapshot failed: ' + (j.error || 'unknown');
    } catch (e) { status.textContent = 'Snapshot failed: ' + e; }
  });
  document.getElementById('feed').addEventListener('error', () => {
    status.textContent = 'Stream interrupted - is the daemon running?';
  });
</script>
</body>
</html>"""


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame (from the SDK) to JPEG bytes (Pillow expects RGB)."""
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB, contiguous for PIL
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _reopen_media(mini: ReachyMini) -> None:
    """Rebuild the camera reader after a GStreamer End-of-stream / stall."""
    try:
        mini.media_manager.close()
    except Exception:
        pass
    backend = getattr(mini, "_media_backend", "default")
    log_level = getattr(mini, "_log_level", "INFO")
    mini.media_manager = mini._configure_mediamanager(backend, log_level)


def _capture_loop(mini: ReachyMini) -> None:
    """Continuously grab frames from the robot camera into a shared buffer.

    Frames arrive at ~30 fps; ``get_frame()`` returns ``None`` between frames,
    which is normal. If frames stall for a few seconds (e.g. a GStreamer
    End-of-stream), the camera reader is rebuilt automatically.
    """
    global _latest_jpeg, _frame_count
    last_good = time.monotonic()
    stalled = False
    next_reopen = 0.0
    while not _stop.is_set():
        try:
            frame = mini.media.get_frame()
        except Exception:
            frame = None
        now = time.monotonic()
        if frame is None:
            # No new frame yet - normal between frames. Only recover on a
            # sustained stall (End-of-stream, daemon pipeline reconfigure).
            if now - last_good > STALL_TIMEOUT:
                if not stalled:
                    print("[camera_app] Camera stalled; attempting to recover...")
                    stalled = True
                if now >= next_reopen:
                    try:
                        _reopen_media(mini)
                    except Exception as exc:
                        print(f"[camera_app] Recovery attempt failed (daemon down?): {exc}")
                    next_reopen = now + STALL_TIMEOUT
                    last_good = now  # give the rebuilt reader time to warm up
            time.sleep(0.005)
            continue
        if stalled:
            print("[camera_app] Camera recovered.")
            stalled = False
        last_good = now
        try:
            jpeg = _encode_jpeg(frame)
        except Exception as exc:
            print(f"[camera_app] Encode error: {exc}")
            time.sleep(0.02)
            continue
        with _latest_lock:
            _latest_jpeg = jpeg
            _frame_count += 1


class Handler(BaseHTTPRequestHandler):
    """Serves the viewer page, the MJPEG stream, and snapshot saving."""

    def log_message(self, *args) -> None:  # silence default request logging
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/stream":
            self._send_stream()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/snapshot":
            self._snapshot()
        else:
            self.send_error(404)

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        period = 1.0 / TARGET_FPS
        last = -1
        try:
            while not _stop.is_set():
                with _latest_lock:
                    jpeg = _latest_jpeg
                    count = _frame_count
                if jpeg is None or count == last:
                    time.sleep(period / 2)
                    continue
                last = count
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # client/browser closed the stream

    def _snapshot(self) -> None:
        with _latest_lock:
            jpeg = _latest_jpeg
        if jpeg is None:
            self._send_json({"ok": False, "error": "no frame yet"})
            return
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        name = "snapshot_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        (SNAPSHOT_DIR / name).write_bytes(jpeg)
        with Image.open(io.BytesIO(jpeg)) as im:
            w, h = im.size
        self._send_json({"ok": True, "file": name, "width": w, "height": h})

    def _send_json(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    display_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    url = f"http://{display_host}:{PORT}"

    print("[camera_app] Connecting to Reachy Mini daemon...")
    try:
        mini = ReachyMini()
    except Exception as exc:
        print(f"[camera_app] Could not connect: {exc}")
        print("[camera_app] Is the daemon running?  Start it with:  reachy-mini-daemon")
        return

    with mini:
        capture = threading.Thread(target=_capture_loop, args=(mini,), daemon=True)
        capture.start()

        server = ThreadingHTTPServer((HOST, PORT), Handler)
        print(f"[camera_app] Camera feed ready at {url}  (Ctrl+C to stop)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[camera_app] Shutting down...")
        finally:
            _stop.set()
            server.shutdown()
            server.server_close()
            capture.join(timeout=2)


if __name__ == "__main__":
    main()
