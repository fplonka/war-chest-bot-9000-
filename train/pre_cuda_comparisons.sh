#!/bin/sh
# Pre-CUDA plan section 7 — offline comparisons on runs/pre_cuda_random/buffer.npz
# Run after the 4.5h dump run finishes. Each step writes its results to
# runs/pre_cuda_random/*.json. Steps are ordered; later steps reuse earlier
# decisions.
set -e
cd "$(dirname "$0")/.."
DUMP=runs/pre_cuda_random/buffer.npz
OUT=runs/pre_cuda_random

# 7a — card/holding screen, on all training solves, one seed each.
python train/offline.py "$DUMP" \
  --arch h384-d64-r64 h384-d64-r64-noe h384-d64-r64-noid h384-d64-r64-nor \
  --steps 4000 --lr 1e-3,3e-4 --seed 7 --out "$OUT/7a_card_holding.json"

# 7b — board/head 2x2 screen: flat vs hex x head 128/384.
python train/offline.py "$DUMP" \
  --arch h768-d128-r64-e64-head128 h768-d128-r64-e64-head384 \
         hex-h768-d128-r64-e64-head128 hex-h768-d128-r64-e64-head384 \
  --steps 4000 --lr 1e-3,3e-4 --seed 7 --out "$OUT/7b_board_head.json"

# 7c — final head-width sweep on the chosen encoder (update after 7b).
python train/offline.py "$DUMP" \
  --arch h768-d128-r64-e64-head64 h768-d128-r64-e64-head128 \
         h768-d128-r64-e64-head192 h768-d128-r64-e64-head384 \
  --steps 4000 --lr 1e-3,3e-4 --seed 7 --out "$OUT/7c_head_width.json"
