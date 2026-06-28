"""Render MonST3R per-frame output as a COLORED 3D orbit (or 4D time playback) mp4.

Rebuilds the point cloud from per-frame color+depth+pose, confidence-filtered
(removes the low-confidence streaks), and renders with a pure-numpy
painter's-algorithm z-buffer (no pyrender / OpenGL needed).

Run on the A100 inside ~/monst3r:
    python render_4d.py --indir demo_tmp/NULL \
        --out /home/azureuser/cloudfiles/code/Users/alstefa/orbit.mp4

Knobs:
    --mode fused   clean orbit of the whole cloud (default)
    --mode time    4D playback: scene animates while camera arcs (--window N)
    --conf-keep 0.6  keep top 60% most-confident points (raise to clean more)
    --arc 70 --frames 120 --fps 30 --width 960 --height 540
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from PIL import Image


def quat_to_R(q):  # [qx,qy,qz,qw]
    x, y, z, w = q
    n = max((x * x + y * y + z * z + w * w) ** 0.5, 1e-9)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_cloud(indir, conf_keep):
    K = np.loadtxt(os.path.join(indir, "pred_intrinsics.txt"))
    K = np.asarray(K).reshape(-1, 3, 3)[0]
    Kinv = np.linalg.inv(K)
    traj = np.atleast_2d(np.loadtxt(os.path.join(indir, "pred_traj.txt")))
    poses = traj[:, 1:]  # tx ty tz qx qy qz qw  (camera-to-world, TUM)
    n = len(poses)

    pts, cols, fidx = [], [], []
    for i in range(n):
        dpath = os.path.join(indir, f"frame_{i}.npy")
        ipath = os.path.join(indir, f"frame_{i}.png")
        cpath = os.path.join(indir, f"conf_{i}.npy")
        if not (os.path.exists(dpath) and os.path.exists(ipath)):
            continue
        depth = np.load(dpath).astype(np.float64)
        rgb = np.asarray(Image.open(ipath).convert("RGB"))
        H, W = depth.shape[:2]
        if rgb.shape[:2] != (H, W):
            rgb = np.asarray(Image.fromarray(rgb).resize((W, H)))
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones_like(uu)
        pix = np.stack([uu, vv, ones], 0).reshape(3, -1).astype(np.float64)
        ray = Kinv @ pix                       # 3 x N, z=1
        cam = ray * depth.reshape(1, -1)       # scale by depth
        t = poses[i, :3]
        R = quat_to_R(poses[i, 3:7])
        world = (R @ cam).T + t                # N x 3
        col = rgb.reshape(-1, 3)

        m = np.isfinite(world).all(1) & (depth.reshape(-1) > 1e-6)
        if os.path.exists(cpath):
            conf = np.load(cpath).astype(np.float64).reshape(-1)
            if conf_keep < 1.0:
                thr = np.quantile(conf[m], 1.0 - conf_keep)
                m &= conf >= thr
        pts.append(world[m]); cols.append(col[m]); fidx.append(np.full(m.sum(), i))
    P = np.concatenate(pts); C = np.concatenate(cols).astype(np.uint8)
    F = np.concatenate(fidx)
    return P, C, F, poses, n


def rotate_about(p, center, axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    v = p - center
    c, s = np.cos(ang), np.sin(ang)
    return center + v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def render_view(P, C, eye, center, up, W, H, f):
    fwd = center - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    upv = np.cross(right, fwd)
    Rwc = np.stack([right, -upv, fwd], 0)      # world->cam
    cam = (P - eye) @ Rwc.T
    z = cam[:, 2]
    good = z > 1e-3
    cam = cam[good]; col = C[good]; z = z[good]
    u = (f * cam[:, 0] / z + W / 2).astype(np.int64)
    v = (f * cam[:, 1] / z + H / 2).astype(np.int64)
    img = np.zeros((H, W, 3), np.uint8)
    order = np.argsort(-z)                      # far -> near (near wins)
    u, v, col = u[order], v[order], col[order]
    for du in (0, 1):                           # 2x2 splat to fill gaps
        for dv in (0, 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = col[m]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="demo_tmp/NULL")
    ap.add_argument("--out", default="orbit.mp4")
    ap.add_argument("--mode", choices=["fused", "time"], default="fused")
    ap.add_argument("--conf-keep", type=float, default=0.6)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--arc", type=float, default=70.0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-points", type=int, default=3000000)
    args = ap.parse_args()

    print("Loading per-frame cloud ...", flush=True)
    P, C, F, poses, nfr = load_cloud(args.indir, args.conf_keep)
    print(f"  {len(P):,} points from {nfr} frames", flush=True)

    # robust center / scale
    center = np.median(P, axis=0)
    d = np.linalg.norm(P - center, axis=1)
    keep = d < np.quantile(d, 0.98)
    P, C, F = P[keep], C[keep], F[keep]
    center = np.median(P, axis=0)

    # start from the FIRST camera's viewpoint so framing is recognizable
    eye0 = poses[0, :3]
    up = quat_to_R(poses[0, 3:7]) @ np.array([0.0, -1.0, 0.0])  # camera up in world
    f = 0.9 * args.width

    if args.mode == "fused" and len(P) > args.max_points:
        idx = np.random.default_rng(0).choice(len(P), args.max_points, replace=False)
        P, C, F = P[idx], C[idx], F[idx]

    half = np.radians(args.arc) / 2.0
    frames = []
    for i in range(args.frames):
        theta = half * np.sin(2 * np.pi * i / args.frames)
        eye = rotate_about(eye0, center, up, theta)
        if args.mode == "time":
            t_in = int(i / args.frames * nfr)
            sel = (F <= t_in) & (F > t_in - args.window)
            Pi, Ci = P[sel], C[sel]
        else:
            Pi, Ci = P, C
        frames.append(render_view(Pi, Ci, eye, center, up,
                                  args.width, args.height, f))
        if (i + 1) % 20 == 0 or i == args.frames - 1:
            print(f"  frame {i + 1}/{args.frames}", flush=True)

    import imageio.v2 as imageio
    try:
        with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None) as w:
            for fr in frames:
                w.append_data(fr)
        print("Wrote", args.out, flush=True)
    except Exception as exc:  # no ffmpeg -> gif
        gif = os.path.splitext(args.out)[0] + ".gif"
        print(f"mp4 failed ({exc}); writing {gif}", flush=True)
        imageio.mimsave(gif, frames, fps=args.fps)
        print("Wrote", gif, flush=True)


if __name__ == "__main__":
    main()
