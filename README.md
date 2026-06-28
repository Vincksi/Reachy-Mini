# Reachy Mini

Application suite for the **Pollen Robotics Reachy Mini Lite** robot (USB connection on COM3).

## Requirements

- Windows 10/11
- Python 3.10+
- Reachy Mini connected via USB (COM3)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install reachy-mini opencv-python pillow numpy mediapipe ultralytics websockets requests psutil
```

## Daemon

The daemon exposes a REST API at `http://127.0.0.1:8000`. All scripts connect to it.

```powershell
python -m reachy_mini.daemon.app.main
```

The daemon must run **continuously** in a separate terminal.

## Video Server (FastAPI / Uvicorn)

```powershell
uvicorn server_video:app --host 0.0.0.0 --port 8000
```

## Scripts

### Camera stream (HTTP server)
```powershell
python capture/camera/camera_app.py
# http://127.0.0.1:8001 - MJPEG stream + snapshots
```

### Hand tracking (MediaPipe)
```powershell
python tracking/hand_tracking.py
```

### Object tracking (YOLO + ByteTrack)
```powershell
python tracking/track_object.py
```

### Manual control (keyboard)
```powershell
python tests_robot/robot_control.py
```

### Room scan (VGGT / 3D)
```powershell
python render_4d/room_scan.py
```

### World model (3D "Tesla" view)
```powershell
python world_model_stuff/world_model.py
```

### Multi-view synchronized capture
```powershell
python capture/multiview_capture.py plan
```

### 4D / orbit rendering
```powershell
python render_4d/render_orbit.py <file.glb>
python render_4d/render_4d.py <folder>
```

## Daemon API

| Endpoint | Description |
|---|---|
| `GET /api/daemon/status` | Daemon status |
| `GET /api/media/status` | Camera availability |
| `POST /api/daemon/restart` | Restart daemon |

API restart can leave the camera in an unstable state — use `room_scan.py` instead, which kills and relaunches the process cleanly.
