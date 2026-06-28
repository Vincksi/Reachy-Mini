# Plan — Reachy Mini Camera Feed App

## Request
"Create a Reachy Mini app that gives the camera feed."

## Context (verified on this machine)
- Robot: **Reachy Mini Lite** (USB, `COM3`); daemon running locally at `http://127.0.0.1:8000`.
- Camera: **arducam**, default **1280×720 @ 30fps**. `/api/media/status` → `available: true`.
- Local camera IPC pipe present: `\\.\pipe\reachymini_camera_pipe`.
- venv: `reachy_mini_env` (Python 3.12). Added **Pillow** for JPEG encoding (no cv2/PIL before).

## Key finding that drove the design
The **JavaScript SDK** (telepresence-style apps) streams via the Hugging Face central
signaling server and is documented as **"supported on wireless versions only"** (HF OAuth
required). This robot is the **Lite / local** variant, so the JS path does not fit.
For a local Lite robot the camera is read directly from the daemon via the **Python SDK
LOCAL backend**: `mini.media.get_frame()` → BGR numpy array `(H, W, 3)`.

## Decision (made autonomously — user was unavailable)
- **Flavour:** standalone **Python app** (same pattern as `hello.py`) — not a JS HF Space,
  not a formal `reachy-mini-app-assistant` scaffold. This is the fastest working camera feed
  for a local Lite robot.
- **Delivery:** serve the feed as an **MJPEG stream over a tiny stdlib HTTP server**, viewable
  in any browser. No GUI library required.
- **Features:** live feed + **snapshot** button (saves a JPEG to `./snapshots/`).

## Technical approach — `camera_app.py`
- Connects with `ReachyMini()` (auto-selects the LOCAL media backend).
- Capture thread: `mini.media.get_frame()` → Pillow JPEG → shared buffer (lock-protected).
- `http.server.ThreadingHTTPServer`:
  - `GET /`         → HTML viewer (mobile-friendly, dark theme).
  - `GET /stream`   → `multipart/x-mixed-replace` MJPEG.
  - `POST /snapshot`→ saves latest frame to `./snapshots/`, returns JSON.
- Binds `127.0.0.1:8001` (localhost only for safety; LAN/phone access is an opt-in noted in code).
- Opens the browser automatically; `Ctrl+C` shuts down cleanly.

## How to run
1. Daemon running: `reachy-mini-daemon` (hardware) — already up.
2. `.\reachy_mini_env\Scripts\activate.ps1`
3. `python camera_app.py` → browser opens at `http://127.0.0.1:8001`.

## Upgrade path (later, if wanted)
- Make it a discoverable on-robot app via `reachy-mini-app-assistant` (Python), or
- a shareable HF Space using the JS telepresence template (requires wireless + HF account).

## Deferred questions (sensible defaults chosen)
- LAN/phone access (bind `0.0.0.0`)? Default **off** for security.
- Snapshot vs record vs head controls? Implemented **snapshot only** (minimal, per request).
