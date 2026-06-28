"""Reachy Mini - full-room pan-tilt scan for 3D reconstruction (VGGT).

Unlike the object-centric modes in multiview_capture.py, this script treats the
robot as a panoramic SCANNER sitting in the middle of a room and sweeps every
reachable degree of freedom to cover as much of the room as possible:

  * body yaw   (azimuth)   - the big one: +/-160 deg => 320 deg horizontal sweep.
                             The joint can't do a full 360, so ~40 deg directly
                             behind the robot stays a blind spot.
  * head pitch (elevation) - look down -> level -> up, to capture floor, walls
                             and ceiling (the vertical "rings" of the panorama).
  * head yaw   (parallax)  - OPTIONAL micro-dither that nudges the camera a few
                             cm sideways per direction. Pure rotation gives weak
                             parallax; a little translation helps VGGT recover
                             depth. Off by default (--head-yaw-steps 1).

It captures a (yaw x pitch [x head-yaw]) grid of overlapping views and writes them
to a FLAT folder you can hand straight to VGGT (images/ holds ONLY .jpg files, so
it won't trip VGGT's image loader):

    captures/<scene>/images/img_0000.jpg ...
    captures/<scene>/metadata.jsonl    # per-frame body_yaw + head angles
    captures/<scene>/manifest.json

QUICK START (REAL hardware: daemon running + robot powered on):

    .\\reachy_mini_env\\Scripts\\python.exe room_scan.py --scene room
    .\\reachy_mini_env\\Scripts\\python.exe room_scan.py --scene room \\
        --yaw-steps 17 --pitch-min -25 --pitch-max 25 --pitch-steps 5
    .\\reachy_mini_env\\Scripts\\python.exe room_scan.py --dry-run   # preview grid only

Cover a WHOLE room by relocating the robot. Each run APPENDS to the same scene
(img_NNNN keeps counting up), and --restart-daemon makes the daemon re-detect the
robot after you replug its USB. So at EVERY new spot, run ONE command:

    .\\reachy_mini_env\\Scripts\\python.exe room_scan.py --scene room --restart-daemon --location spotA
    # move + replug the robot, then:
    .\\reachy_mini_env\\Scripts\\python.exe room_scan.py --scene room --restart-daemon --location spotB
    # ...repeat at each spot. All frames pool into captures/room/images/.

Then reconstruct on the A100 (same flow as the bottle):

    Compress-Archive -Path captures\\room\\images\\* -DestinationPath room_images.zip -Force
    # upload room_images.zip to the A100, then in ~/vggt_run/vggt:
    #   python demo_gradio.py   -> open the gradio.live URL, upload images, Reconstruct

Requires: reachy-mini daemon on REAL hardware (sim has no camera) + Pillow/numpy.
ALWAYS run with the venv python:  .\\reachy_mini_env\\Scripts\\python.exe
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# --- Config / verified hardware facts -------------------------------------
DEFAULT_PORT = 8000
YAW_LIMIT_RAD = 2.7925  # body_yaw hard limit +/-160 deg (urdf joint "yaw_body")
JPEG_QUALITY = 95
CAPTURE_ROOT = Path(__file__).parent / "captures"
WARMUP_TIMEOUT = 30.0  # camera pipeline can take a while to stream after a restart

_stop = threading.Event()


# --- Helpers --------------------------------------------------------------
def _encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame (from the SDK) to JPEG bytes (Pillow expects RGB)."""
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _conn_kwargs(args: argparse.Namespace) -> dict:
    """Build ReachyMini() connection kwargs from CLI args."""
    kw: dict = {"port": args.port, "connection_mode": args.connection_mode}
    if args.host:
        kw["host"] = args.host
    return kw


def _restart_daemon(host: Optional[str], port: int, timeout: float) -> None:
    """Fully restart the daemon PROCESS so the camera pipeline is rebuilt.

    IMPORTANT: the daemon's API restart (/api/daemon/restart) re-inits the motors
    but leaves the GStreamer CAMERA pipeline stalled -- media reports 'available'
    yet get_frame() returns None forever. Only a full process restart fixes the
    camera, so we kill the running daemon and relaunch it fresh.
    """
    base_host = host or "127.0.0.1"
    # 1) kill any running daemon process (matched by its entry point in cmdline)
    try:
        import psutil  # provided by reachy-mini
        me = os.getpid()
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "reachy-mini-daemon" in cmd and proc.pid != me:
                    print(f"[room_scan] stopping daemon PID {proc.pid}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as exc:
        print(f"[room_scan] could not stop daemon ({exc}); restart it manually.")

    # 2) wait for port 8000 to free, then relaunch the daemon in its own console
    end = time.monotonic() + 10
    while time.monotonic() < end and not _port_free(base_host, port):
        time.sleep(0.5)
    exe = Path(__file__).parent / "reachy_mini_env" / "Scripts" / "reachy-mini-daemon.exe"
    if exe.exists():
        print(f"[room_scan] launching fresh daemon: {exe.name}")
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen([str(exe)], creationflags=flags,
                             cwd=str(Path(__file__).parent))
        except Exception as exc:
            print(f"[room_scan] failed to launch daemon ({exc}); start it manually.")
    else:
        print(f"[room_scan] daemon exe not found ({exe}); "
              "start `reachy-mini-daemon` manually in another terminal.")

    # 3) wait until the daemon is running AND the camera is delivering
    _wait_daemon_camera(base_host, port, timeout)


def _port_free(host: str, port: int) -> bool:
    """True if nothing is listening on host:port yet."""
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        s.close()
        return False
    except OSError:
        return True


def _wait_daemon_camera(host: str, port: int, timeout: float) -> None:
    """Poll until daemon state == 'running' AND camera media is available."""
    import requests  # provided by reachy-mini
    base = f"http://{host}:{port}"
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if _stop.is_set():
            return
        try:
            state = requests.get(f"{base}/api/daemon/status",
                                 timeout=5).json().get("state")
        except Exception:
            state = "starting"
        if state != last:
            print(f"[room_scan] daemon state: {state}")
            last = state
        media_ok = False
        if state == "running":
            try:
                media_ok = bool(requests.get(f"{base}/api/media/status",
                                             timeout=5).json().get("available"))
            except Exception:
                media_ok = False
        if state == "running" and media_ok:
            print("[room_scan] daemon running + camera available.")
            time.sleep(1.5)  # small grace for the video pipeline to start streaming
            return
        time.sleep(1.0)
    print("[room_scan] Daemon/camera not fully ready in time; connecting anyway.")


def _warmup(mini: ReachyMini, timeout: float = WARMUP_TIMEOUT) -> np.ndarray:
    """Wait for the camera to deliver its first frame; raise if none arrives."""
    deadline = time.monotonic() + timeout
    warned = False
    while time.monotonic() < deadline:
        if _stop.is_set():
            raise KeyboardInterrupt
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        if not warned:
            print("[room_scan] Waiting for camera frames "
                  "(daemon up? camera works on REAL hardware only, not --sim)...")
            warned = True
        time.sleep(0.1)
    raise RuntimeError(
        f"No camera frames within {timeout:.0f}s. Check the daemon / real hardware."
    )


# --- Output writer --------------------------------------------------------
class RoomWriter:
    """Writes frames into captures/<scene>/images/ (flat, VGGT-ready).

    images/ holds ONLY .jpg files; per-frame metadata + manifest go in the parent
    scene folder so they never reach VGGT's image loader.
    """

    def __init__(self, scene_dir: Path, manifest: dict) -> None:
        self.images_dir = scene_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Resume numbering after any existing img_*.jpg so re-running at a NEW
        # robot location APPENDS to the same scene instead of overwriting it.
        nums = [int(p.stem[4:]) for p in self.images_dir.glob("img_*.jpg")
                if p.stem[4:].isdigit()]
        self.start = (max(nums) + 1) if nums else 0
        self.n = self.start
        # append (not truncate) so earlier locations' metadata survives
        self.meta_f = (scene_dir / "metadata.jsonl").open("a", encoding="utf-8")
        self.manifest_path = scene_dir / "manifest.json"
        self.manifest = manifest
        self.w: Optional[int] = None
        self.h: Optional[int] = None
        if self.start:
            print(f"[room_scan] Resuming scene at img_{self.start:04d} "
                  f"({self.start} image(s) already present).")

    def write(self, frame_bgr: np.ndarray, extra: dict) -> int:
        idx = self.n
        fname = f"img_{idx:04d}.jpg"
        (self.images_dir / fname).write_bytes(_encode_jpeg(frame_bgr))
        self.h, self.w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        rec = {"idx": idx, "file": fname, "t_wall": time.time()}
        rec.update(extra)
        self.meta_f.write(json.dumps(rec) + "\n")
        self.meta_f.flush()
        self.n += 1
        return idx

    def close(self) -> None:
        added = self.n - self.start
        self.manifest.update({
            "num_frames_total": self.n,
            "num_frames_added": added,
            "frame_width": self.w,
            "frame_height": self.h,
            "finished_iso": datetime.now(timezone.utc).isoformat(),
        })
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        self.meta_f.close()
        print(f"[room_scan] Added {added} image(s); {self.n} total "
              f"-> {self.images_dir}")


# --- Scan grid ------------------------------------------------------------
def _build_grid(args: argparse.Namespace):
    """Return (yaw_rads, pitches_deg, head_yaws_deg) sample arrays."""
    yaw_limit = min(abs(args.yaw_range), np.degrees(YAW_LIMIT_RAD))
    yaws = np.radians(np.linspace(-yaw_limit, yaw_limit, max(1, args.yaw_steps)))
    pitches = np.linspace(args.pitch_min, args.pitch_max, max(1, args.pitch_steps))
    if args.head_yaw_steps > 1 and args.head_yaw > 0:
        head_yaws = np.linspace(-args.head_yaw, args.head_yaw, args.head_yaw_steps)
    else:
        head_yaws = np.array([0.0])
    return yaws, pitches, head_yaws


def _print_plan(yaws, pitches, head_yaws, args: argparse.Namespace) -> None:
    n = len(yaws) * len(pitches) * len(head_yaws)
    per = args.move_duration + args.settle
    print("\n[room_scan] Scan grid:")
    print(f"  body yaw  : {len(yaws)} steps  "
          f"{np.degrees(yaws[0]):+.0f}..{np.degrees(yaws[-1]):+.0f} deg (azimuth)")
    print(f"  head pitch: {len(pitches)} steps  "
          f"{pitches[0]:+.0f}..{pitches[-1]:+.0f} deg (elevation)")
    if len(head_yaws) > 1:
        print(f"  head yaw  : {len(head_yaws)} steps  "
              f"{head_yaws[0]:+.0f}..{head_yaws[-1]:+.0f} deg (parallax)")
    print(f"  => {n} frames, ~{n * per:.0f}s total (~{per:.1f}s/frame). "
          "Note: ~40 deg behind the robot is a blind spot.\n")


def do_scan(args: argparse.Namespace) -> None:
    """Sweep body-yaw x head-pitch (x head-yaw) and capture a frame per pose."""
    yaws, pitches, head_yaws = _build_grid(args)
    _print_plan(yaws, pitches, head_yaws, args)
    if args.dry_run:
        print("[room_scan] --dry-run: no hardware used, no frames captured.")
        return

    scene_dir = CAPTURE_ROOT / args.scene
    manifest = {
        "scene": args.scene,
        "mode": "room_scan",
        "location": args.location,
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "yaw_range_deg": float(np.degrees(yaws[-1])),
        "yaw_steps": len(yaws),
        "pitch_min_deg": args.pitch_min,
        "pitch_max_deg": args.pitch_max,
        "pitch_steps": len(pitches),
        "head_yaw_deg": args.head_yaw,
        "head_yaw_steps": len(head_yaws),
        "settle": args.settle,
        "move_duration": args.move_duration,
        "host": args.host or "localhost",
        "port": args.port,
        "connection_mode": args.connection_mode,
        "jpeg_quality": JPEG_QUALITY,
        "sdk_frame_format": "BGR from SDK, saved as RGB JPEG",
    }

    if args.restart_daemon:
        _restart_daemon(args.host, args.port, args.daemon_timeout)

    with ReachyMini(**_conn_kwargs(args)) as mini:
        _warmup(mini, timeout=args.warmup)
        try:
            mini.enable_motors()
            # CRITICAL: the SDK default automatic_body_yaw=True makes the IK
            # recompute the body-rotation joint from the head pose (yaw=0 here),
            # so the explicit body_yaw is ignored and the base never spins. Turn
            # it OFF so our body_yaw targets actually rotate the robot on its axis.
            mini.set_automatic_body_yaw(False)
        except Exception as exc:  # real robot needs power; surface but continue
            print(f"[room_scan] Warning: motor/body-yaw setup failed ({exc}).")
        writer = RoomWriter(scene_dir, manifest)
        try:
            for col, yaw in enumerate(yaws):
                if _stop.is_set():
                    break
                # serpentine: alternate pitch direction each column -> less travel
                col_pitches = pitches if col % 2 == 0 else pitches[::-1]
                for pitch in col_pitches:
                    for hy in head_yaws:
                        if _stop.is_set():
                            break
                        pose = create_head_pose(
                            pitch=float(pitch), yaw=float(hy), degrees=True
                        )
                        try:
                            mini.goto_target(body_yaw=float(yaw), head=pose,
                                             duration=args.move_duration)
                        except Exception as exc:
                            print(f"[room_scan] move yaw={np.degrees(yaw):+.0f} "
                                  f"pitch={pitch:+.0f} failed: {exc}")
                            continue
                        time.sleep(args.settle)  # settle to avoid motion blur
                        frame = mini.media.get_frame()
                        if frame is None:
                            continue
                        idx = writer.write(frame, {
                            "loc": args.location,
                            "body_yaw": float(yaw),
                            "body_yaw_deg": float(np.degrees(yaw)),
                            "head_pitch_deg": float(pitch),
                            "head_yaw_deg": float(hy),
                        })
                        print(f"[room_scan] yaw={np.degrees(yaw):+6.1f}  "
                              f"pitch={pitch:+5.1f}  hy={hy:+4.1f}  -> img {idx}")
        finally:
            try:  # return home
                mini.goto_target(body_yaw=0.0, head=create_head_pose(),
                                 duration=args.move_duration)
            except Exception:
                pass
            writer.close()

    _print_next_steps(args, scene_dir)


def _print_next_steps(args: argparse.Namespace, scene_dir: Path) -> None:
    print("\n[room_scan] Next: reconstruct the room with VGGT on the A100.")
    print(f"  1) zip:    Compress-Archive -Path {scene_dir}\\images\\* "
          f"-DestinationPath {args.scene}_images.zip -Force")
    print(f"  2) upload {args.scene}_images.zip to the A100 (Jupyter/VS Code file tree)")
    print("  3) in ~/vggt_run/vggt:  python demo_gradio.py")
    print("     -> open the public gradio.live URL, upload the images, click Reconstruct.")
    print(f"  (sanity-check locally first:  python validate_capture.py {scene_dir})")


# --- CLI ------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Reachy Mini full-room pan-tilt scan for VGGT 3D reconstruction."
    )
    p.add_argument("--scene", default="room", help="Scene name under captures/.")
    p.add_argument("--yaw-range", dest="yaw_range", type=float, default=160.0,
                   help="Body-yaw half-range in degrees (clamped to 160).")
    p.add_argument("--yaw-steps", dest="yaw_steps", type=int, default=13,
                   help="Number of azimuth stops across the yaw range.")
    p.add_argument("--pitch-min", dest="pitch_min", type=float, default=-22.0,
                   help="Lowest head pitch (look down) in degrees.")
    p.add_argument("--pitch-max", dest="pitch_max", type=float, default=22.0,
                   help="Highest head pitch (look up) in degrees.")
    p.add_argument("--pitch-steps", dest="pitch_steps", type=int, default=3,
                   help="Number of elevation rings.")
    p.add_argument("--head-yaw", dest="head_yaw", type=float, default=0.0,
                   help="Optional head-yaw dither amplitude (deg) for extra parallax.")
    p.add_argument("--head-yaw-steps", dest="head_yaw_steps", type=int, default=1,
                   help="Head-yaw samples per direction (1 = off).")
    p.add_argument("--settle", type=float, default=0.4,
                   help="Seconds to settle after each move (reduce motion blur).")
    p.add_argument("--move-duration", dest="move_duration", type=float, default=0.6)
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print the scan grid and exit (no hardware, no capture).")
    p.add_argument("--restart-daemon", dest="restart_daemon", action="store_true",
                   help="Fully restart the daemon PROCESS (kill + relaunch) so it "
                        "re-detects a replugged robot AND rebuilds the camera "
                        "pipeline (the API restart leaves the camera stalled). "
                        "Run at each new spot after moving the robot.")
    p.add_argument("--daemon-timeout", dest="daemon_timeout", type=float, default=30.0,
                   help="Seconds to wait for the daemon to report 'running' after restart.")
    p.add_argument("--warmup", type=float, default=WARMUP_TIMEOUT,
                   help="Seconds to wait for the first camera frame before giving up.")
    p.add_argument("--location", default=None,
                   help="Optional label for this spot, tagged into each frame's metadata.")
    # connection
    p.add_argument("--host", default=None,
                   help="Daemon host (default: SDK default / localhost).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--connection-mode", dest="connection_mode", default="auto",
                   choices=["auto", "localhost_only", "network"])

    args = p.parse_args(argv)
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    try:
        do_scan(args)
    except KeyboardInterrupt:
        print("\n[room_scan] Interrupted (partial capture saved).")
        return 130
    except Exception as exc:
        print(f"[room_scan] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
