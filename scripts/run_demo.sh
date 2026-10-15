#!/usr/bin/env bash
# PPMem single-image demo: one image plus a camera trajectory -> a generated video.
#
#   ./scripts/run_demo.sh --image path/to.png --extrinsic path/to/extrinsic.npy --out runs/demo
#
# --out must not hold output from an earlier run; add --overwrite to delete it first.
# A run that failed before producing anything leaves only its log, which is fine.
#
# === Prepare the weights and dependencies described in README ===
#
#   PPMem defaults to the Hugging Face model Robbyant-Research/PPMem.
#   MOGE_SRC      Optional MoGe v2 source checkout to put on PYTHONPATH. Leave unset
#                 when installed through requirements.txt. MoGe estimates the first
#                 frame's depth and intrinsics; --init_geometry bypasses it.
#   compiled      flash_attn_interface (FlashAttention-3) and diff_rect_rasterization.
#                 MagiAttention is optional; see README for installation.
#
# === Optional ===
#   --checkpoint <directory or HF repo ID> overrides the default PPMem model.
#   --moge-checkpoint <file or HF repo ID> overrides the default MoGe model.
#   --init_geometry <x.npz> supply frame 0's depth and intrinsics instead of estimating
#                           them. np.savez(path, depth=(H,W) float32 metres,
#                           fx=, fy=, cx=, cy=) -- all four in pixels at that depth's resolution.
#   PPMEM_CONFIG            alternative config file (default configs/inference.yaml)
#   CUDA_VISIBLE_DEVICES    pick a GPU
#
#   Offline: use local model paths, or prime the HF cache and set HF_HUB_OFFLINE=1.
#   MOGE_SRC selects source code; --moge-checkpoint selects weights.
#
# Any other argument is passed straight through to `ppmem`; see `ppmem --help`.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=${PPMEM_CONFIG:-$REPO/configs/inference.yaml}
OUT=""; IMAGE=""; EXTRINSIC=""; START_STEP=0; INIT_GEOM=""; USER_CFG=""; CHECKPOINT=""; MOGE_CHECKPOINT=""; OVERWRITE=0; PASS=()
die() { echo "🔴 $*" >&2; exit 1; }

# Both `--flag value` and `--flag=value`.
while [ $# -gt 0 ]; do
  _a=$1; _v=""
  case "$_a" in --*=*) _v=${_a#*=}; _a=${_a%%=*}; _eq=1 ;; *) _eq=0 ;; esac
  _take() {  # Read an inline value or the next argument.
    if [ "$_eq" = 1 ]; then printf '%s' "$_v"
    else [ $# -ge 2 ] || die "$_a needs a value"; printf '%s' "$2"; fi
  }
  case "$_a" in
    --out|--output_root) OUT=$(_take "$@"); [ "$_eq" = 1 ] && shift || shift 2 ;;
    --image)             IMAGE=$(_take "$@"); [ "$_eq" = 1 ] && shift || shift 2 ;;
    --extrinsic)         EXTRINSIC=$(_take "$@"); [ "$_eq" = 1 ] && shift || shift 2 ;;
    --start_step)        START_STEP=$(_take "$@"); PASS+=("--start_step" "$START_STEP")
                         [ "$_eq" = 1 ] && shift || shift 2 ;;
    --init_geometry)     INIT_GEOM=$(_take "$@"); PASS+=("--init_geometry" "$INIT_GEOM")
                         [ "$_eq" = 1 ] && shift || shift 2 ;;
    --checkpoint) CHECKPOINT=$(_take "$@"); [ -n "$CHECKPOINT" ] || die "--checkpoint needs a value"; PASS+=("--checkpoint" "$CHECKPOINT")
                         [ "$_eq" = 1 ] && shift || shift 2 ;;
    --moge-checkpoint) MOGE_CHECKPOINT=$(_take "$@"); [ -n "$MOGE_CHECKPOINT" ] || die "--moge-checkpoint needs a value"; PASS+=("--moge-checkpoint" "$MOGE_CHECKPOINT")
                         [ "$_eq" = 1 ] && shift || shift 2 ;;
    --config)            USER_CFG=$(_take "$@"); [ "$_eq" = 1 ] && shift || shift 2 ;;
    --overwrite)         [ "$_eq" = 0 ] || die "--overwrite is a switch and takes no value (got $1). Leave it off to keep $OUT."
                         OVERWRITE=1; shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
# load_config takes the last --config, so a user's goes after ours.
[ -n "$USER_CFG" ] && CFG_EXTRA=(--config "$USER_CFG") || CFG_EXTRA=()


# Resolve before the rm -rf below, so a missing interpreter does not cost the
# previous run. PEP 394: `python` may not exist.
PY=""
for _c in python3 python; do command -v "$_c" >/dev/null 2>&1 && { PY="$_c"; break; }; done
[ -n "$PY" ] || die "no python3 (or python) on PATH"
[ -n "$IMAGE" ]     || die "need --image <input image>"
[ -n "$EXTRINSIC" ] || die "need --extrinsic <.npy with (N,4,4) c2w poses>"
[ -n "$OUT" ]       || die "need --out <output directory>"
[ -f "$IMAGE" ]     || die "image not found: $IMAGE"
[ -f "$EXTRINSIC" ] || die "trajectory not found: $EXTRINSIC"
[ -f "$CFG" ]       || die "config not found: $CFG (set PPMEM_CONFIG to point elsewhere)"
if [ -n "${MOGE_SRC:-}" ]; then
  [ -d "$MOGE_SRC/moge" ] || die "MOGE_SRC looks wrong: no moge/ under $MOGE_SRC"
fi

# An optional source checkout overrides the installed MoGe. Point at <MoGe>
# itself, not its parent: a stale utils3d there can shadow the installed package.
export PYTHONPATH="$REPO${MOGE_SRC:+:$MOGE_SRC}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# MoGe's DINOv2 backbone falls back to torch SDPA without xformers.
export XFORMERS_DISABLED=${XFORMERS_DISABLED:-1}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Cap the thread count when running several jobs on one machine: torch sizes its
# intra-op pool by core count, and the resulting oversubscription can make the CPU
# stages more than twice as slow (the GPU stages are unaffected).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# Validate explicit/local model paths before any output cleanup. Hub downloads happen in Python.
"$PY" - "$CHECKPOINT" "$MOGE_CHECKPOINT" "$INIT_GEOM" "$START_STEP" <<'PY_MODELS'
import sys
from ppmem.checkpoints import checkpoint_source, moge_source
try:
    if sys.argv[1]:
        checkpoint_source(sys.argv[1])
    if not sys.argv[3] and int(sys.argv[4]) <= 0:
        moge_source(sys.argv[2] or None)
except (ValueError, FileNotFoundError) as e:
    sys.exit(f"🔴 {e}")
PY_MODELS

# Check with the same interpreter and environment as inference, before cleanup.
# --init_geometry supplies frame 0's geometry and does not need MoGe.
if [ -z "$INIT_GEOM" ] && [ "$START_STEP" -le 0 ]; then
  "$PY" -c 'from moge.model.v2 import MoGeModel' || \
    die "cannot import MoGe v2; prepare the environment as described in README, then run in the PPMem repository: $PY -m pip install --no-build-isolation -r requirements.txt. Alternatively set MOGE_SRC=<MoGe v2 checkout>. See the import error above. --init_geometry does not need MoGe."
fi

# --start_step N (N>0) preserves the existing pool and requires its directory.
if [ "$START_STEP" -gt 0 ]; then
  [ -d "$OUT/pool" ] || die "--start_step $START_STEP resumes an existing pool, but $OUT/pool does not exist"
  echo "Resuming from step $START_STEP, keeping $OUT/pool"
  exec_resume=1
else
  exec_resume=0
fi

# `Pipeline finished.` is the completion marker; the mp4 exists while it is still
# being written.
if [ "$exec_resume" = 0 ] && [ -f "$OUT/run.log" ] \
   && grep -q 'Pipeline finished\.' "$OUT/run.log" 2>/dev/null; then
  echo "Already finished (run.log has 'Pipeline finished.'), skipping: $OUT"
  exit 0
fi
# Fresh runs need an output directory without earlier output, to avoid loading stale
# pool state. A run that stopped before producing anything leaves only run.log,
# run_meta*.json and empty directories; those do not count, so the same command can
# simply be rerun. --overwrite permits clearing partial output; completed runs were
# skipped above.
if [ "$exec_resume" = 0 ] && [ -d "$OUT" ] \
   && [ -n "$(find "$OUT" ! -type d ! -name run.log ! -name 'run_meta*.json' -print -quit 2>/dev/null)" ]; then
  [ "$OVERWRITE" = 1 ] || die "$OUT holds output from an earlier run. Use a new --out, pass --overwrite to replace it, or pass --start_step to resume the existing run."
  # Before overwrite, check path candidates from arguments and environment paths,
  # including the value after the first "=" in --flag=value.
  # Compare absolute paths and resolved targets against both forms of OUT:
  # a link under OUT is deleted even if its target is outside, and an outside link
  # may point into OUT.
  _abspath()  { "$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))'  "$1" 2>/dev/null; }
  _realpath() { "$PY" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1" 2>/dev/null; }
  _o_abs=$(_abspath "$OUT"); _o_real=$(_realpath "$OUT")
  for _t in "$IMAGE" "$EXTRINSIC" "$INIT_GEOM" "$USER_CFG" "$CFG" \
            "${MOGE_SRC:-}" \
            "${HF_HOME:-}" "${HF_HUB_CACHE:-}" \
            ${PASS[@]+"${PASS[@]}"}; do
    for _f in "$_t" "${_t#*=}"; do
      [ -n "$_f" ] || continue
      _f=$("$PY" -c 'import os,sys; print(os.path.expanduser(sys.argv[1]))' "$_f")
      [ -e "$_f" ] || [ -L "$_f" ] || continue
      for _p in "$(_abspath "$_f")" "$(_realpath "$_f")"; do
        [ -n "$_p" ] || continue
        case "$_p" in
          "$_o_abs"|"$_o_abs"/*|"$_o_real"|"$_o_real"/*)
            die "input lives inside the output directory ($_f -> $_p); --overwrite wipes $OUT and would take it along. Move it out." ;;
        esac
      done
    done
  done
  echo "Removing previous partial run: $OUT"
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

# Single GPU. Pick which one with CUDA_VISIBLE_DEVICES, or pass --device cuda:N.

# Log to $OUT/run.log. -u prevents stdout buffering through tee; -a preserves resume logs.
# pipefail propagates Python failures through the logging pipeline.
"$PY" -u -m ppmem.cli \
  --config "$CFG" \
  --image "$IMAGE" \
  --extrinsic "$EXTRINSIC" \
  --output_root "$OUT" \
  ${PASS[@]+"${PASS[@]}"} ${CFG_EXTRA[@]+"${CFG_EXTRA[@]}"} 2>&1 | tee -a "$OUT/run.log"
