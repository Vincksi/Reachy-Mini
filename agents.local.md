# agents.local.md — session context for AI agents

> Per the Reachy Mini AGENTS.md, read this first. Concise, factual notes for future sessions.

## Robot
- Type: **Lite** (USB). Serial port: **COM3**.
- Daemon runs locally on this laptop at `http://127.0.0.1:8000`.
  - Real hardware: `reachy-mini-daemon`
  - Simulation (MuJoCo, **no camera**): `reachy-mini-daemon --sim`
- Camera: **arducam**, default 1280x720@30fps. Local IPC pipe: `\\.\pipe\reachymini_camera_pipe`.

## Environment (Windows, no admin)
- Python only inside venv **`reachy_mini_env`** (Python 3.12). No system Python (MS Store alias).
  - Activate: `.\reachy_mini_env\Scripts\activate.ps1`  (NOT `source .../bin/activate`)
- Package manager: **`uv`** at `C:\Users\b-nirmiger\.local\bin`. If `uv` is not found in a
  terminal, call it by full path: `& "C:\Users\b-nirmiger\.local\bin\uv.exe" ...`
- `scoop` installed for system tools (git, git-lfs). Use `scoop install`, not `sudo apt`.
- Extra dep added: **pillow** (JPEG encoding for `camera_app.py`).

## Apps in this workspace
- `hello.py`      — minimal demo (wiggles antennas).
- `camera_app.py` — live camera feed -> browser MJPEG at `http://127.0.0.1:8001` + snapshot
                    button (saves to `./snapshots/`). Run: activate venv, then
                    `python camera_app.py` (daemon must be running).

## Notes / gotchas
- JS SDK (telepresence) = wireless + HF OAuth only -> not used for this local Lite robot.
  Camera is read via the Python SDK LOCAL backend: `mini.media.get_frame()` -> **BGR** numpy.
- Camera works on **real hardware only** (sim has no camera).
- After installing a CLI tool, new terminals can have a stale PATH; fully restart VS Code.
- Code 43 "Device Descriptor Request Failed" on USB = bad/charge-only cable; use a data cable.
