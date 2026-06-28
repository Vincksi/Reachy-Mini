"""Render a colored point-cloud GLB (e.g. MonST3R scene.glb) as an orbit video.

Usage:
    .\\reachy_mini_env\\Scripts\\python.exe render_orbit.py --glb scene.glb --out orbit.mp4

Common tweaks:
    --frames 240 --fps 30        length / smoothness
    --arc 360                    full turntable; use e.g. --arc 140 for a there-and-back
                                 sweep when the back of the scene is hollow
    --elev 18                    camera height in degrees
    --up y|z|-y|-z               which data axis points "up" (try -y if upside down)
    --point-size 2.5             dot size (GL renderer only)
    --width 1280 --height 720    output resolution

Needs: trimesh, imageio, imageio-ffmpeg (+ pyrender for the nicer GL render).
    uv pip install trimesh pyrender imageio imageio-ffmpeg pillow
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def load_points(path):
    """Load a GLB/PLY/OBJ and return (points Nx3 float, colors Nx3 uint8)."""
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    all_pts, all_cols = [], []
    for name, geom in scene.geometry.items():
        verts = np.asarray(geom.vertices, dtype=np.float64)
        if verts.size == 0:
            continue
        # apply this geometry's transform from the scene graph
        try:
            T = scene.graph.get(name)[0]
            verts = trimesh.transformations.transform_points(verts, T)
        except Exception:
            pass

        cols = None
        if hasattr(geom, "colors") and geom.colors is not None and len(geom.colors):
            cols = np.asarray(geom.colors)
        elif getattr(geom, "visual", None) is not None:
            vc = getattr(geom.visual, "vertex_colors", None)
            if vc is not None and len(vc):
                cols = np.asarray(vc)
        if cols is None or len(cols) != len(verts):
            cols = np.full((len(verts), 4), 200, dtype=np.uint8)
        cols = cols[:, :3].astype(np.uint8)

        all_pts.append(verts)
        all_cols.append(cols)

    if not all_pts:
        raise SystemExit(f"No point/vertex data found in {path}")
    return np.concatenate(all_pts), np.concatenate(all_cols)


def remap_up(points, up):
    """Rotate points so the chosen data axis becomes world +Y (up)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if up == "y":
        return points
    if up == "-y":
        return np.stack([x, -y, -z], axis=1)
    if up == "z":
        return np.stack([x, z, -y], axis=1)
    if up == "-z":
        return np.stack([x, -z, y], axis=1)
    return points


def look_at(eye, target, up):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def orbit_angles(n, arc_deg):
    """Yield azimuth angles (radians). Full 360 loops; <360 pings back and forth."""
    if arc_deg >= 360:
        return np.linspace(0, 2 * np.pi, n, endpoint=False)
    half = np.radians(arc_deg) / 2.0
    return half * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))


def render_pyrender(pts, cols, args, center, radius, azimuths, phi):
    import pyrender

    scene = pyrender.Scene(bg_color=[8, 8, 12, 255], ambient_light=[1, 1, 1])
    cloud = pyrender.Mesh.from_points(pts, colors=cols)
    scene.add(cloud)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(args.width, args.height, point_size=args.point_size)

    frames = []
    up = np.array([0.0, 1.0, 0.0])
    for theta in azimuths:
        eye = center + radius * np.array(
            [np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)]
        )
        scene.set_pose(cam_node, look_at(eye, center, up))
        color, _ = r.render(scene)
        frames.append(color[:, :, :3])
    r.delete()
    return frames


def render_matplotlib(pts, cols, args, center, radius, azimuths, phi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # subsample for speed
    maxp = max(1000, args.max_points)
    if len(pts) > maxp:
        idx = np.random.default_rng(0).choice(len(pts), maxp, replace=False)
        pts, cols = pts[idx], cols[idx]
    c = cols.astype(np.float32) / 255.0

    elev_deg = np.degrees(phi)
    dpi = 100
    figsize = (args.width / dpi, args.height / dpi)

    # Build the figure + scatter ONCE; only the camera angle changes per frame.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=c, s=args.point_size, marker=".",
               linewidths=0, depthshade=False)
    ax.set_axis_off()
    ax.set_facecolor("#08080c")
    fig.patch.set_facecolor("#08080c")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    frames = []
    n = len(azimuths)
    for i, theta in enumerate(azimuths):
        ax.view_init(elev=elev_deg, azim=np.degrees(theta))
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(img.copy())
        if (i + 1) % 15 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}", flush=True)
    plt.close(fig)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="scene.glb")
    ap.add_argument("--out", default="orbit.mp4")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--arc", type=float, default=360.0)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--up", choices=["y", "-y", "z", "-z"], default="y")
    ap.add_argument("--point-size", type=float, default=2.5)
    ap.add_argument("--max-points", type=int, default=60000,
                    help="cap points for the matplotlib renderer (speed)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--radius", type=float, default=2.2,
                    help="orbit radius as a multiple of scene size")
    args = ap.parse_args()

    print(f"Loading {args.glb} ...")
    pts, cols = load_points(args.glb)
    pts = remap_up(pts, args.up)

    # robust centering: drop far outliers so the framing isn't ruined by stray points
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    keep = d < np.percentile(d, 98)
    pts, cols = pts[keep], cols[keep]
    center = pts.mean(axis=0)
    extent = np.linalg.norm(pts.max(0) - pts.min(0))
    radius = args.radius * (extent / 2.0 + 1e-6)
    phi = np.radians(args.elev)
    azimuths = orbit_angles(args.frames, args.arc)
    print(f"{len(pts):,} points | center={np.round(center,2)} | extent={extent:.2f}")

    try:
        print("Rendering with pyrender (GL) ...")
        frames = render_pyrender(pts, cols, args, center, radius, azimuths, phi)
    except Exception as exc:  # noqa: BLE001 - want any failure to fall back
        print(f"pyrender unavailable ({exc}). Falling back to matplotlib.")
        frames = render_matplotlib(pts, cols, args, center, radius, azimuths, phi)

    import imageio.v2 as imageio

    print(f"Encoding {len(frames)} frames -> {args.out} ...")
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    print(f"Done: {args.out}  ({len(frames)/args.fps:.1f}s @ {args.fps}fps)")


if __name__ == "__main__":
    sys.exit(main())
