"""Reachy Mini - synchronized multi-robot multiview capture (multi-laptop rig).

This is a *capture agent*: you run ONE copy on each laptop, where each laptop is
connected to one Reachy Mini and runs its own `reachy-mini-daemon`. The agent
grabs frames from its LOCAL daemon (`mini.media.get_frame()` - the local camera
backend only works on the same machine as the daemon), timestamps every frame,
and saves them. The agents are synchronized by a shared wall-clock start
time (`--start-at`), so all robots fire together.

Why this design (verified against the installed SDK + agents.local.md):
  - The camera is read via the Python SDK LOCAL backend; reading a *remote*
    robot's camera needs the WebRTC/telepresence path (not used here). So each
    robot must be captured on its own host -> one agent per laptop.
  - `ReachyMini()` auto-connects to the local daemon at 127.0.0.1:8000.
  - Body rotation joint `yaw_body` range is +/-2.7925 rad (+/-160 deg) - used by
    `sweep` mode (from descriptions/reachy_mini/urdf/robot.urdf).

Rig layout (recommended for 3D assets): space the robots evenly in a ring aimed
at a central subject (~0.4-0.8 m) - 3 robots ~120 deg apart, 4 robots ~90 deg -
giving full 360 deg azimuth with real baselines. (3 synchronized views is exactly
the Cosmos AgiBot robot-multiview shape.) NOTE: a robot spinning in place only
PANS its view (the camera is ~5 cm off the spin axis) - it does NOT orbit the
object, so each robot only sees the side facing it. Fill gaps with `fan` mode at
each station (head pitch x yaw wiggle = real micro-baselines + up/down peek) and,
where possible, by nudging the robots between shots - not by spinning. For a
Cosmos-style multiview video sample, place them on a frontal arc.

------------------------------------------------------------------------------
QUICK START (run on ONE laptop to coordinate, then paste per-laptop commands):

    # 1) sync the laptops' clocks first (Windows):  w32tm /resync
    # 2) print a shared start time + the per-laptop commands:
    python multiview_capture.py plan --mode record --fps 15 --duration 5 --in 15

    # 3) paste the printed command on each laptop (cam0..camN). They start together.

Standalone / single-robot test (no sync needed):
    python multiview_capture.py snap   --cam-id cam0 --scene mug
    python multiview_capture.py record --cam-id cam0 --scene mug --fps 15 --duration 5
    python multiview_capture.py sweep  --cam-id cam0 --scene mug --steps 48
    python multiview_capture.py fan    --cam-id cam0 --scene mug   # head light-field
    python multiview_capture.py positions --scene mug --fan        # one robot, many spots

Output (per laptop, local):
    captures/<scene>/<cam-id>/frame_000000.jpg ...
    captures/<scene>/<cam-id>/metadata.jsonl   # idx, file, t_wall, t_mono_ns, body_yaw
    captures/<scene>/<cam-id>/manifest.json    # scene/cam/mode/params/camera info

After capture, copy each laptop's `captures/<scene>/<cam-id>/` into one shared
`captures/<scene>/` folder, then feed to reconstruction (VGGT/COLMAP -> gsplat):
  - object 3D: run `collect` to pool all pos*/ + cam*/ frames into images/, then
    one VGGT/COLMAP run on that folder.
  - Cosmos multiview: keep each cam as a separate view (pinhole_<view>).

Requires: the daemon running on REAL hardware (sim has no camera), and Pillow.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import signal
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
JPEG_QUALITY = 95  # high quality: fewer compression artifacts for SfM/3DGS
CAPTURE_ROOT = Path(__file__).parent / "captures"
WARMUP_TIMEOUT = 10.0  # seconds to wait for the first camera frame

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


def _wait_until(start_at: Optional[float]) -> None:
    """Block until the shared wall-clock start time (for cross-laptop sync)."""
    if not start_at:
        return
    remaining = start_at - time.time()
    if remaining > 0:
        print(f"[multiview] Waiting {remaining:.2f}s for synchronized start "
              f"(epoch {start_at:.3f})...")
    while not _stop.is_set() and time.time() < start_at:
        time.sleep(0.001)  # tight spin near the deadline for low jitter


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
            print("[multiview] Waiting for camera frames "
                  "(daemon up? camera only works on REAL hardware, not --sim)...")
            warned = True
        time.sleep(0.1)
    raise RuntimeError(
        "No camera frames received within "
        f"{timeout:.0f}s. Check the daemon and that this is real hardware."
    )


# --- Output writer --------------------------------------------------------
class CamWriter:
    """Writes one camera's frames + per-frame metadata + a manifest."""

    def __init__(self, scene_dir: Path, cam_id: str, manifest: dict) -> None:
        self.cam_dir = scene_dir / cam_id
        self.cam_dir.mkdir(parents=True, exist_ok=True)
        self.meta_f = (self.cam_dir / "metadata.jsonl").open("w", encoding="utf-8")
        self.manifest_path = self.cam_dir / "manifest.json"
        self.manifest = manifest
        self.n = 0
        self.w: Optional[int] = None
        self.h: Optional[int] = None

    def write(self, frame_bgr: np.ndarray, body_yaw: Optional[float] = None,
              extra: Optional[dict] = None) -> int:
        idx = self.n
        fname = f"frame_{idx:06d}.jpg"
        (self.cam_dir / fname).write_bytes(_encode_jpeg(frame_bgr))
        self.h, self.w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        rec = {
            "idx": idx,
            "file": fname,
            "t_wall": time.time(),
            "t_mono_ns": time.monotonic_ns(),
            "body_yaw": body_yaw,
        }
        if extra:
            rec.update(extra)
        self.meta_f.write(json.dumps(rec) + "\n")
        self.meta_f.flush()
        self.n += 1
        return idx

    def close(self) -> None:
        self.manifest.update({
            "num_frames": self.n,
            "frame_width": self.w,
            "frame_height": self.h,
            "finished_iso": datetime.now(timezone.utc).isoformat(),
        })
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        self.meta_f.close()
        print(f"[multiview] Wrote {self.n} frame(s) -> {self.cam_dir}")


def _base_manifest(args: argparse.Namespace, mode: str) -> dict:
    return {
        "scene": args.scene,
        "cam_id": args.cam_id,
        "mode": mode,
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "start_at": args.start_at,
        "host": args.host or "localhost",
        "port": args.port,
        "connection_mode": args.connection_mode,
        "jpeg_quality": JPEG_QUALITY,
        "sdk_frame_format": "BGR from SDK, saved as RGB JPEG",
    }


# --- Capture modes --------------------------------------------------------
def do_snap(args: argparse.Namespace) -> None:
    """Capture a single (optionally synchronized) frame."""
    scene_dir = CAPTURE_ROOT / args.scene
    with ReachyMini(**_conn_kwargs(args)) as mini:
        fallback = _warmup(mini)
        _wait_until(args.start_at)
        if _stop.is_set():
            return
        frame = mini.media.get_frame()
        if frame is None:
            frame = fallback
        writer = CamWriter(scene_dir, args.cam_id, _base_manifest(args, "snap"))
        writer.write(frame)
        writer.close()


def do_record(args: argparse.Namespace) -> None:
    """Capture a synchronized burst at --fps for --duration seconds (4D stream)."""
    if args.fps <= 0 or args.duration <= 0:
        raise ValueError("--fps and --duration must be positive.")
    scene_dir = CAPTURE_ROOT / args.scene
    manifest = _base_manifest(args, "record")
    manifest.update({"fps": args.fps, "duration": args.duration})
    period = 1.0 / args.fps
    n_target = int(round(args.fps * args.duration))
    with ReachyMini(**_conn_kwargs(args)) as mini:
        _warmup(mini)
        _wait_until(args.start_at)
        writer = CamWriter(scene_dir, args.cam_id, manifest)
        t0 = time.monotonic()
        dropped = 0
        for i in range(n_target):
            if _stop.is_set():
                break
            target = t0 + i * period
            dt = target - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            frame = mini.media.get_frame()
            if frame is None:
                dropped += 1
                continue
            writer.write(frame)
        if dropped:
            print(f"[multiview] Note: {dropped} empty frame(s) skipped.")
        writer.close()


def do_sweep(args: argparse.Namespace) -> None:
    """Rotate the body through a yaw sweep, capturing a frame at each step.

    NOTE: spinning the body PANS the camera (it sits only ~5 cm off the spin
    axis), so this samples view DIRECTIONS, not viewpoints around an object - use
    it for panoramic scene coverage and for keeping a subject framed, NOT to
    capture the sides of an object (the robots' ring positions do that). The yaw
    angle is saved per frame as a pose prior.
    """
    scene_dir = CAPTURE_ROOT / args.scene
    yaw_min = max(-YAW_LIMIT_RAD, args.yaw_min)
    yaw_max = min(YAW_LIMIT_RAD, args.yaw_max)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    angles = np.linspace(yaw_min, yaw_max, args.steps)
    manifest = _base_manifest(args, "sweep")
    manifest.update({
        "steps": args.steps, "yaw_min": yaw_min, "yaw_max": yaw_max,
        "settle": args.settle, "move_duration": args.move_duration,
    })
    with ReachyMini(**_conn_kwargs(args)) as mini:
        _warmup(mini)
        try:
            mini.enable_motors()
        except Exception as exc:  # real robot needs power; surface but continue
            print(f"[multiview] Warning: enable_motors() failed ({exc}).")
        _wait_until(args.start_at)
        writer = CamWriter(scene_dir, args.cam_id, manifest)
        for ang in angles:
            if _stop.is_set():
                break
            try:
                mini.goto_target(body_yaw=float(ang), duration=args.move_duration)
            except Exception as exc:
                print(f"[multiview] Move to yaw={ang:+.3f} failed: {exc}")
                continue
            time.sleep(args.settle)  # let the head settle to avoid motion blur
            frame = mini.media.get_frame()
            if frame is None:
                continue
            idx = writer.write(frame, body_yaw=float(ang))
            print(f"[multiview] {args.cam_id} yaw={ang:+.3f} rad "
                  f"({np.degrees(ang):+6.1f} deg) -> frame {idx}")
        try:
            mini.goto_target(body_yaw=0.0, duration=args.move_duration)  # home
        except Exception:
            pass
        writer.close()


def _capture_head_fan(mini, writer, *, pitch_range: float, pitch_steps: int,
                      yaw_range: float, yaw_steps: int, settle: float,
                      move_duration: float, label: str = "") -> None:
    """Sweep the head over a pitch x yaw grid, capturing a frame at each pose."""
    pitches = np.linspace(-pitch_range, pitch_range, pitch_steps)
    yaws = np.linspace(-yaw_range, yaw_range, yaw_steps)
    for pitch in pitches:
        for yaw in yaws:
            if _stop.is_set():
                return
            pose = create_head_pose(pitch=float(pitch), yaw=float(yaw), degrees=True)
            try:
                mini.goto_target(head=pose, duration=move_duration)
            except Exception as exc:
                print(f"[multiview] Head pitch={pitch:+.0f} yaw={yaw:+.0f} failed: {exc}")
                continue
            time.sleep(settle)
            frame = mini.media.get_frame()
            if frame is None:
                continue
            idx = writer.write(frame, extra={
                "head_pitch_deg": float(pitch), "head_yaw_deg": float(yaw),
            })
            print(f"[multiview] {label} head pitch={pitch:+5.1f} yaw={yaw:+5.1f} "
                  f"deg -> frame {idx}")
    try:
        mini.goto_target(head=create_head_pose(), duration=move_duration)  # home
    except Exception:
        pass


def do_fan(args: argparse.Namespace) -> None:
    """Capture a light-field "fan" from ONE fixed robot station.

    Sweeps the 6-DOF head over a grid of pitch x yaw angles (the head TRANSLATES
    a few cm and tilts, unlike body-spin which only pans), capturing a frame at
    each. This squeezes a cluster of slightly-different viewpoints - with real
    (small) baselines and some up/down peek - out of one fixed position. Run it
    on each of the 3 robots; pool all stations -> sparse-view reconstruction
    (InstantSplat / VGGT) can build an asset from just 3 wiggling robots.
    """
    if args.pitch_steps < 1 or args.yaw_steps < 1:
        raise ValueError("--pitch-steps and --yaw-steps must be >= 1.")
    scene_dir = CAPTURE_ROOT / args.scene
    manifest = _base_manifest(args, "fan")
    manifest.update({
        "pitch_range_deg": args.pitch_range, "pitch_steps": args.pitch_steps,
        "yaw_range_deg": args.yaw_range, "yaw_steps": args.yaw_steps,
        "settle": args.settle, "move_duration": args.move_duration,
    })
    with ReachyMini(**_conn_kwargs(args)) as mini:
        _warmup(mini)
        try:
            mini.enable_motors()
        except Exception as exc:
            print(f"[multiview] Warning: enable_motors() failed ({exc}).")
        _wait_until(args.start_at)
        writer = CamWriter(scene_dir, args.cam_id, manifest)
        _capture_head_fan(mini, writer, pitch_range=args.pitch_range,
                          pitch_steps=args.pitch_steps, yaw_range=args.yaw_range,
                          yaw_steps=args.yaw_steps, settle=args.settle,
                          move_duration=args.move_duration, label=args.cam_id)
        writer.close()


def _next_position_index(scene_dir: Path, prefix: str) -> int:
    """Next free position index, so re-running does not overwrite existing folders."""
    if not scene_dir.is_dir():
        return 0
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = [int(m.group(1)) for d in scene_dir.iterdir()
            if d.is_dir() and (m := pat.match(d.name))]
    return max(nums) + 1 if nums else 0


def do_positions(args: argparse.Namespace) -> None:
    """Interactive single-robot capture from several positions (static scene).

    For validating reconstruction while the rig is offline: place ONE robot,
    press ENTER to capture, move it around the object, repeat. Because the scene
    is static, these sequential positions are equivalent to that many cameras.
    Each position is saved to its own folder (pos00, pos01, ...). With --fan,
    each position also does a head pitch x yaw cluster for extra views.
    """
    scene_dir = CAPTURE_ROOT / args.scene
    print("\n[multiview] Multi-position capture (one robot).")
    print("[multiview] Keep the OBJECT and LIGHTING still; move the ROBOT around it.")
    print("[multiview] Press ENTER to capture each position; type 'q' + ENTER to finish.\n")
    idx = _next_position_index(scene_dir, args.prefix)  # resume; never overwrite
    if idx:
        print(f"[multiview] Resuming at {args.prefix}{idx:02d} "
              f"(existing {args.prefix} folders found).\n")
    with ReachyMini(**_conn_kwargs(args)) as mini:
        _warmup(mini)
        if args.fan:
            try:
                mini.enable_motors()
            except Exception as exc:
                print(f"[multiview] Warning: enable_motors() failed ({exc}).")
        while not _stop.is_set():
            try:
                resp = input(f"  position {idx:02d}  [ENTER=capture, q=finish]: ")
            except (EOFError, KeyboardInterrupt):
                break
            if resp.strip().lower() == "q":
                break
            cam_id = f"{args.prefix}{idx:02d}"
            manifest = _base_manifest(args, "positions")
            manifest["cam_id"] = cam_id
            manifest["position_index"] = idx
            writer = CamWriter(scene_dir, cam_id, manifest)
            if args.fan:
                _capture_head_fan(mini, writer, pitch_range=args.pitch_range,
                                  pitch_steps=args.pitch_steps, yaw_range=args.yaw_range,
                                  yaw_steps=args.yaw_steps, settle=args.settle,
                                  move_duration=args.move_duration, label=cam_id)
            else:
                got = 0
                for _ in range(max(1, args.shots)):
                    frame = mini.media.get_frame()
                    if frame is not None:
                        writer.write(frame)
                        got += 1
                    time.sleep(args.shot_interval)
                if got == 0:
                    print("  [warn] no frames here (is the camera delivering frames?).")
            writer.close()
            idx += 1
    print(f"\n[multiview] Done: {idx} position(s) -> {scene_dir}")
    if idx:
        print(f"[multiview] Validate:  python validate_capture.py {scene_dir}")
        print("[multiview] Then pool all frames and run VGGT/COLMAP on the A100.")


def do_collect(args: argparse.Namespace) -> None:
    """Pool every captured frame (pos*/, cam*/, ...) into one images/ folder.

    Reconstruction tools (VGGT/COLMAP) want all images of a scene in a single
    folder. This copies every frame_*.jpg under captures/<scene>/ into
    captures/<scene>/<out>/ with sequential names, and writes a manifest mapping
    each pooled file back to its source (position/camera) for later pose work.
    """
    scene_dir = CAPTURE_ROOT / args.scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene not found: {scene_dir}")
    out_dir = scene_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [p for p in sorted(scene_dir.rglob("frame_*.jpg"))
            if out_dir not in p.parents]
    if not srcs:
        print(f"[multiview] No frame_*.jpg found under {scene_dir}.")
        return
    mapping = {}
    for i, src in enumerate(srcs):
        dst_name = f"img_{i:04d}.jpg"
        shutil.copy2(src, out_dir / dst_name)
        mapping[dst_name] = str(src.relative_to(scene_dir)).replace("\\", "/")
    (out_dir / "collect_manifest.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    print(f"[multiview] Pooled {len(srcs)} image(s) -> {out_dir}")
    print("[multiview] Reconstruct on the A100 (Linux). Point the tool at this folder:")
    print(f"    # VGGT (fast, sparse-view): copy {args.out}/ to vggt/scene/images/, then")
    print("    #   python demo_colmap.py --scene_dir=scene/ --use_ba")
    print("    #   pip install gsplat==1.3.0")
    print("    #   python examples/simple_trainer.py default --data_dir scene/ --result_dir out/")
    print(f"    # Dense (COLMAP+3DGS): ns-process-data images --data {args.out} --output-dir proc")
    print("    #                      ns-train splatfacto --data proc")


def do_plan(args: argparse.Namespace) -> None:
    """Print a shared start time + ready-to-paste commands for each laptop."""
    start_at = time.time() + args.delay
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    human = datetime.fromtimestamp(start_at).strftime("%H:%M:%S")
    if args.mode == "record":
        tail = f"record --fps {args.fps:g} --duration {args.duration:g}"
    elif args.mode == "sweep":
        tail = f"sweep --steps {args.steps}"
    else:
        tail = "snap"
    print(f"\n[multiview] Synchronized start epoch: {start_at:.3f}  "
          f"(~{human} local, in {args.delay:g}s)")
    print(f"[multiview] Current epoch:            {time.time():.3f}")
    print("[multiview] First run `w32tm /resync` on every laptop so clocks agree.\n")
    print("Paste ONE command on each laptop (same --scene), they fire together:\n")
    for cam in cams:
        print(f"  # ---- {cam} ----")
        print(f"  python multiview_capture.py {tail} "
              f"--cam-id {cam} --scene {args.scene} --start-at {start_at:.3f}\n")


# --- CLI ------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=None,
                   help="Daemon host (default: SDK default / localhost).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--connection-mode", dest="connection_mode", default="auto",
                   choices=["auto", "localhost_only", "network"])
    p.add_argument("--scene", default="scene",
                   help="Scene name (use the SAME on all laptops).")
    p.add_argument("--cam-id", dest="cam_id", default="cam0",
                   help="Per-robot id, e.g. cam0..cam3.")
    p.add_argument("--start-at", dest="start_at", type=float, default=None,
                   help="Unix epoch to begin (for cross-laptop sync).")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reachy Mini multi-robot multiview capture agent "
                    "(run one per laptop)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snap", help="Capture one (synchronized) frame.")
    _add_common(p_snap)
    p_snap.set_defaults(func=do_snap)

    p_rec = sub.add_parser("record", help="Synchronized burst at --fps for --duration.")
    _add_common(p_rec)
    p_rec.add_argument("--fps", type=float, default=15.0)
    p_rec.add_argument("--duration", type=float, default=5.0)
    p_rec.set_defaults(func=do_record)

    p_sw = sub.add_parser("sweep", help="Body-yaw sweep capture (static object).")
    _add_common(p_sw)
    p_sw.add_argument("--steps", type=int, default=48)
    p_sw.add_argument("--yaw-min", dest="yaw_min", type=float, default=-YAW_LIMIT_RAD)
    p_sw.add_argument("--yaw-max", dest="yaw_max", type=float, default=YAW_LIMIT_RAD)
    p_sw.add_argument("--settle", type=float, default=0.4,
                      help="Seconds to settle after each move (reduce blur).")
    p_sw.add_argument("--move-duration", dest="move_duration", type=float, default=0.5)
    p_sw.set_defaults(func=do_sweep)

    p_fan = sub.add_parser("fan",
                           help="Head pitch x yaw light-field from one fixed station.")
    _add_common(p_fan)
    p_fan.add_argument("--pitch-range", dest="pitch_range", type=float, default=20.0,
                       help="Head pitch up/down extent in degrees (+/-).")
    p_fan.add_argument("--pitch-steps", dest="pitch_steps", type=int, default=3)
    p_fan.add_argument("--yaw-range", dest="yaw_range", type=float, default=20.0,
                       help="Head yaw left/right extent in degrees (+/-).")
    p_fan.add_argument("--yaw-steps", dest="yaw_steps", type=int, default=3)
    p_fan.add_argument("--settle", type=float, default=0.4,
                       help="Seconds to settle after each move (reduce blur).")
    p_fan.add_argument("--move-duration", dest="move_duration", type=float, default=0.5)
    p_fan.set_defaults(func=do_fan)

    p_pos = sub.add_parser("positions",
                           help="Interactive: one robot, capture from many positions.")
    _add_common(p_pos)
    p_pos.add_argument("--prefix", default="pos",
                       help="Position folder prefix (pos00, pos01, ...).")
    p_pos.add_argument("--shots", type=int, default=1,
                       help="Frames to grab per position (snap mode).")
    p_pos.add_argument("--shot-interval", dest="shot_interval", type=float, default=0.15)
    p_pos.add_argument("--fan", action="store_true",
                       help="At each position, also do a head pitch x yaw cluster.")
    p_pos.add_argument("--pitch-range", dest="pitch_range", type=float, default=20.0)
    p_pos.add_argument("--pitch-steps", dest="pitch_steps", type=int, default=3)
    p_pos.add_argument("--yaw-range", dest="yaw_range", type=float, default=20.0)
    p_pos.add_argument("--yaw-steps", dest="yaw_steps", type=int, default=3)
    p_pos.add_argument("--settle", type=float, default=0.4)
    p_pos.add_argument("--move-duration", dest="move_duration", type=float, default=0.5)
    p_pos.set_defaults(func=do_positions)

    p_col = sub.add_parser("collect",
                           help="Pool all pos*/ + cam*/ frames into one images/ folder.")
    p_col.add_argument("--scene", default="scene", help="Scene under captures/ to pool.")
    p_col.add_argument("--out", default="images", help="Output subfolder name.")
    p_col.set_defaults(func=do_collect)

    p_plan = sub.add_parser("plan", help="Print a sync start time + per-laptop commands.")
    p_plan.add_argument("--in", dest="delay", type=float, default=10.0,
                        help="Seconds until the synchronized start.")
    p_plan.add_argument("--cams", default="cam0,cam1,cam2",
                        help="Comma-separated robot ids (one per laptop).")
    p_plan.add_argument("--scene", default="scene")
    p_plan.add_argument("--mode", default="record", choices=["snap", "record", "sweep"])
    p_plan.add_argument("--fps", type=float, default=15.0)
    p_plan.add_argument("--duration", type=float, default=5.0)
    p_plan.add_argument("--steps", type=int, default=48)
    p_plan.set_defaults(func=do_plan)

    args = parser.parse_args(argv)
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[multiview] Interrupted.")
        return 130
    except Exception as exc:
        print(f"[multiview] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
