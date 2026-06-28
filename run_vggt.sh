#!/usr/bin/env bash
# run_vggt.sh — reconstruct a 3D scene from a folder of images with VGGT, on a
# CUDA GPU box (e.g. your A100). Clones VGGT, sets up a venv, and runs it.
#
# Usage (on the GPU box):
#   chmod +x run_vggt.sh
#   ./run_vggt.sh <images_dir> [mode]
#       mode = colmap (default) | viser | gradio
#
#   colmap : export camera poses + 3D points to <scene>/sparse/  (feeds gsplat)
#   viser  : interactive 3D point-cloud viewer in the browser
#   gradio : web UI like the HF Space (auto-creates a public *.gradio.live link)
#
# gradio launches with share=True -> open the public URL it prints (no forwarding).
# viser serves on a local port; reach it via SSH forward or the Azure ML app proxy:
#   ssh -L 8080:localhost:8080  <user>@<gpu-ip>
#
set -euo pipefail

IMAGES="${1:-./images}"
MODE="${2:-colmap}"
WORKDIR="${WORKDIR:-$HOME/vggt_run}"
REPO="https://github.com/facebookresearch/vggt"

if [ ! -d "$IMAGES" ]; then
  echo "[run_vggt] ERROR: images folder not found: $IMAGES" >&2
  exit 1
fi
N=$(find "$IMAGES" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "[run_vggt] $N image(s) in '$IMAGES', mode=$MODE"
if [ "$N" -lt 2 ]; then echo "[run_vggt] Need >= 2 images." >&2; exit 1; fi

# 1) clone VGGT
mkdir -p "$WORKDIR"; cd "$WORKDIR"
if [ ! -d vggt ]; then
  echo "[run_vggt] cloning VGGT..."
  git clone --depth 1 "$REPO"
fi
cd vggt

# 2) python env (isolated; remove the venv block to use the system/conda env)
if [ ! -d .venv ]; then
  # VGGT needs Python >= 3.10; pick the newest available interpreter
  # (Azure ML's default azureml_py38 env is Python 3.8 and will NOT work).
  PYBIN=""
  for cand in python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
      PYBIN="$cand"; break
    fi
  done
  if [ -z "$PYBIN" ]; then
    echo "[run_vggt] ERROR: VGGT needs Python >= 3.10 but only $(python3 -V 2>&1) is on PATH." >&2
    echo "[run_vggt] On Azure ML run:  conda create -y -n vggt python=3.11 && conda activate vggt  then re-run." >&2
    exit 1
  fi
  echo "[run_vggt] creating venv with $PYBIN ($("$PYBIN" -V 2>&1))"
  "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_demo.txt
# requirements_demo.txt (via lightglue) upgrades numpy to 2.x, which breaks
# torch 2.3.1 ("compiled using NumPy 1.x ..."). Pin numpy back to the 1.x line.
pip install "numpy<2"

# 3) stage images into scene/images/
SCENE="$WORKDIR/scene"
mkdir -p "$SCENE/images"
shopt -s nullglob nocaseglob
cp -f "$IMAGES"/*.jpg "$IMAGES"/*.jpeg "$IMAGES"/*.png "$SCENE/images/" 2>/dev/null || true
shopt -u nullglob nocaseglob
echo "[run_vggt] staged $(find "$SCENE/images" -type f | wc -l) image(s) -> $SCENE/images"

# 4) run
case "$MODE" in
  colmap)
    # Feed-forward reconstruction is the robust default: VGGT predicts poses +
    # depth directly, so it works on sparse/low-texture scenes where classical
    # bundle adjustment fails ("Not enough inliers"). Opt into BA with USE_BA=1
    # only for dense, well-textured captures.
    BA_FLAG=""
    if [ "${USE_BA:-0}" = "1" ]; then BA_FLAG="--use_ba"; fi
    if [ -n "$BA_FLAG" ]; then
      echo "[run_vggt] running feed-forward reconstruction + bundle adjustment..."
    else
      echo "[run_vggt] running feed-forward reconstruction (no BA; robust for sparse views)..."
    fi
    python demo_colmap.py --scene_dir="$SCENE" $BA_FLAG
    echo
    echo "[run_vggt] DONE -> COLMAP model at: $SCENE/sparse/"
    echo "[run_vggt] To make a Gaussian splat:"
    echo "    cd $WORKDIR && git clone https://github.com/nerfstudio-project/gsplat"
    echo "    cd gsplat && pip install -e ."
    echo "    python examples/simple_trainer.py default --data_factor 1 \\"
    echo "        --data_dir $SCENE --result_dir $WORKDIR/out"
    ;;
  viser)
    echo "[run_vggt] starting viser viewer — open the printed URL (forward the port first)."
    python demo_viser.py --image_folder "$SCENE/images"
    ;;
  gradio)
    echo "[run_vggt] starting gradio app — it launches with share=True, so open the"
    echo "[run_vggt] public https://<id>.gradio.live URL it prints (no port-forwarding needed)."
    python demo_gradio.py
    ;;
  *)
    echo "[run_vggt] Unknown mode '$MODE' (use: colmap | viser | gradio)" >&2
    exit 1
    ;;
esac
