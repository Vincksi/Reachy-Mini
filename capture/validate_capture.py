"""validate_capture.py - is a capture good enough to reconstruct a 3D asset?

Run this WHILE the robots are being synchronized: reconstruction quality depends
only on the images, not on sync, so you can de-risk the whole 3D pipeline now
with a single robot (or a phone walk-around). If one camera's orbit reconstructs,
the 3-robot rig will too.

Usage:
  python validate_capture.py captures/<scene>          # scene (all cam*/ dirs)
  python validate_capture.py captures/<scene>/cam0     # one camera folder
  python validate_capture.py --images <folder>         # any folder of images

Checks (CPU only, no GPU): image count, resolution consistency, sharpness (blur),
brightness/exposure consistency, texture (feature richness), and angular coverage
(read from metadata.jsonl `body_yaw` if present). If OpenCV is installed it also
estimates frame-to-frame feature-match connectivity, which predicts whether
Structure-from-Motion (COLMAP / VGGT) will link the images.
  Enable the stronger check:  uv pip install opencv-python-headless

Then upload the images to the VGGT Hugging Face Space to get an actual 3D point
cloud + camera poses with no local GPU:
  https://huggingface.co/spaces/facebook/vggt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2  # optional; enables feature-match connectivity estimate

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAXW = 512  # downscale width for metric computation (speed)

# Heuristic thresholds (intentionally conservative; tune for your scenes).
MIN_IMAGES_OK = 30
MIN_IMAGES_WARN = 15
BLUR_REL = 0.4  # frame sharpness < BLUR_REL * median => flagged blurry
DARK_LUMA = 30  # mean luma below this => underexposed
BRIGHT_LUMA = 225  # mean luma above this => blown out
TEXTURE_MIN = 2.0  # mean gradient magnitude floor for a "textured" frame
EXPO_STD_WARN = 35.0  # std of per-frame brightness across the set
GAP_WARN_DEG = 40.0  # largest angular gap between yaw samples
SPAN_WARN_DEG = 120.0  # total yaw span covered
CONN_WARN = 50  # median good feature matches between adjacent frames


# --- per-image metrics ----------------------------------------------------
def _load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)))
    return np.asarray(im, dtype=np.float32)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian - higher = sharper, low = blurry."""
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _texture(gray: np.ndarray) -> float:
    """Mean gradient magnitude - proxy for feature richness / matchability."""
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    m = min(gx.shape[0], gy.shape[0])
    n = min(gx.shape[1], gy.shape[1])
    return float(np.hypot(gx[:m, :n], gy[:m, :n]).mean())


# --- discovery ------------------------------------------------------------
def _find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _find_cam_dirs(scene: Path) -> list[Path]:
    return sorted(
        d for d in scene.iterdir()
        if d.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in d.iterdir())
    )


def _coverage(folder: Path) -> Optional[dict]:
    """Angular coverage from metadata.jsonl body_yaw, if available."""
    meta = folder / "metadata.jsonl"
    if not meta.exists():
        return None
    yaws = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        try:
            y = json.loads(line).get("body_yaw")
        except json.JSONDecodeError:
            continue
        if y is not None:
            yaws.append(float(y))
    if len(yaws) < 2:
        return None
    ys = np.degrees(np.sort(np.asarray(yaws)))
    gaps = np.diff(ys)
    return {"span_deg": float(ys[-1] - ys[0]),
            "max_gap_deg": float(gaps.max()), "n": len(yaws)}


def _connectivity(images: list[Path]) -> list[int]:
    """Good ORB matches between consecutive frames (needs OpenCV)."""
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    counts: list[int] = []
    prev_des = None
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, des = orb.detectAndCompute(img, None)
        if prev_des is not None and des is not None:
            good = 0
            for pair in bf.knnMatch(prev_des, des, k=2):
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                    good += 1
            counts.append(good)
        prev_des = des
    return counts


# --- analysis -------------------------------------------------------------
def analyze_folder(folder: Path, label: str, min_images: int) -> str:
    images = _find_images(folder)
    n = len(images)
    print(f"\n=== {label} ===")
    print(f"  images: {n}")
    if n == 0:
        print("  [FAIL] no images found.")
        return "FAIL"

    sharps, brights, texts, sizes = [], [], [], []
    for p in images:
        g = _load_gray(p)
        sharps.append(_sharpness(g))
        brights.append(g.mean())
        texts.append(_texture(g))
        sizes.append(Image.open(p).size)
    sharps = np.asarray(sharps)
    brights = np.asarray(brights)
    texts = np.asarray(texts)

    issues: list[str] = []
    warns: list[str] = []

    # count
    if n < min_images // 2:
        issues.append(f"only {n} images; aim for >= {min_images} for a clean asset.")
    elif n < min_images:
        warns.append(f"{n} images is light; >= {min_images} gives better coverage.")

    # resolution consistency
    uniq = sorted(set(sizes))
    if len(uniq) == 1:
        print(f"  resolution: {uniq[0][0]}x{uniq[0][1]} (consistent)")
    else:
        warns.append(f"mixed resolutions {uniq[:3]}... - keep one camera/size.")

    # blur
    med_sharp = float(np.median(sharps))
    blurry = [images[i].name for i in range(n) if sharps[i] < BLUR_REL * med_sharp]
    print(f"  sharpness: median {med_sharp:.0f} (higher=sharper)")
    if blurry:
        warns.append(f"{len(blurry)} blurry frame(s), e.g. {blurry[:3]} "
                     "- increase --settle / better lighting.")

    # exposure
    print(f"  brightness: mean {brights.mean():.0f}, std {brights.std():.0f} (luma 0-255)")
    if brights.std() > EXPO_STD_WARN:
        warns.append("exposure varies a lot across frames - lock camera exposure.")
    if int((brights < DARK_LUMA).sum()):
        warns.append(f"{int((brights < DARK_LUMA).sum())} very dark frame(s).")
    if int((brights > BRIGHT_LUMA).sum()):
        warns.append(f"{int((brights > BRIGHT_LUMA).sum())} blown-out frame(s).")

    # texture / feature richness
    med_tex = float(np.median(texts))
    print(f"  texture: median {med_tex:.1f} (gradient mag; higher=more features)")
    if med_tex < TEXTURE_MIN:
        issues.append("scene looks textureless/low-feature - SfM will likely fail. "
                      "Add a textured backdrop or a patterned mat.")
    elif int((texts < TEXTURE_MIN).sum()):
        warns.append(f"{int((texts < TEXTURE_MIN).sum())} low-texture frame(s).")

    # coverage (from yaw metadata)
    cov = _coverage(folder)
    if cov:
        print(f"  yaw coverage: span {cov['span_deg']:.0f} deg, "
              f"largest gap {cov['max_gap_deg']:.0f} deg")
        if cov["span_deg"] < SPAN_WARN_DEG:
            warns.append(f"yaw span only {cov['span_deg']:.0f} deg - add views from "
                         "more angles (more robots / rotate the object).")
        if cov["max_gap_deg"] > GAP_WARN_DEG:
            warns.append(f"{cov['max_gap_deg']:.0f} deg gap between views - "
                         "capture more steps (--steps).")

    # connectivity (optional, OpenCV)
    if _HAS_CV2 and n >= 2:
        counts = _connectivity(images)
        if counts:
            med_c = float(np.median(counts))
            print(f"  connectivity: median {med_c:.0f} good matches between "
                  "adjacent frames")
            if med_c < CONN_WARN:
                issues.append(f"weak frame-to-frame matching ({med_c:.0f}) - views "
                              "too far apart or too blurry; SfM may not link them.")
    elif not _HAS_CV2:
        print("  connectivity: (skipped - install opencv-python-headless to enable)")

    # verdict
    if issues:
        verdict = "FAIL"
    elif warns:
        verdict = "WARN"
    else:
        verdict = "PASS"
    print(f"  --> {verdict}")
    for msg in issues:
        print(f"      [FAIL] {msg}")
    for msg in warns:
        print(f"      [warn] {msg}")
    return verdict


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate a capture for 3D-asset reconstruction (CPU only)."
    )
    ap.add_argument("path", nargs="?", help="scene dir, camera dir, or image folder.")
    ap.add_argument("--images", help="explicit folder of images to check.")
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES_OK)
    args = ap.parse_args(argv)

    target = Path(args.images or args.path or ".").resolve()
    if not target.exists():
        print(f"[validate] path not found: {target}", file=sys.stderr)
        return 2

    if not _HAS_CV2:
        print("[validate] note: OpenCV not installed - the feature-match "
              "connectivity check (best SfM predictor) is disabled.\n"
              "           enable it: uv pip install opencv-python-headless")

    # Scene with cam*/ subfolders, a single folder, or an image folder.
    verdicts: list[str] = []
    if args.images is None:
        cam_dirs = _find_cam_dirs(target) if target.is_dir() else []
    else:
        cam_dirs = []

    if cam_dirs:
        total = 0
        for d in cam_dirs:
            verdicts.append(analyze_folder(d, f"{target.name}/{d.name}", args.min_images))
            total += len(_find_images(d))
        print(f"\n[validate] scene total: {total} images across {len(cam_dirs)} "
              "camera(s).")
        print("[validate] For object reconstruction, pool ALL cam*/ images into one "
              "VGGT/COLMAP run; for Cosmos multiview keep each cam as a separate view.")
    else:
        verdicts.append(analyze_folder(target, target.name, args.min_images))

    overall = "FAIL" if "FAIL" in verdicts else "WARN" if "WARN" in verdicts else "PASS"
    print(f"\n[validate] OVERALL: {overall}")
    if overall != "FAIL":
        print("[validate] Next: upload the images to "
              "https://huggingface.co/spaces/facebook/vggt to get a 3D point cloud "
              "+ camera poses (no local GPU). A clean ring of poses = your data is good.")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
