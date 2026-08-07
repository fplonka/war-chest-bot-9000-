# pre_cuda_random — the 4.5-hour random-draft dump run (pre-CUDA plan section 6)

## What we were trying

Create a representative random-draft replay dump after the format freeze, so
the section-7 offline comparisons (card/holding, flat-vs-hex, head width) run
on frozen, solve-aligned data. The run's Elo is not a decision metric — the
dump and the 1,000 solver roots are the deliverables.

This is also the first random-draft training run ever: the draft pool now
includes the Warrior Priest pair, so the run doubles as the first long-run
exercise of the pending-coin machinery.

## Recipe

    python train/train.py --minutes 270 --warm-minutes 5 --random-draft \
      --depth 2 --iters 64 --hidden 384 --dg 64 --rank 64 --policy 0 --aux 0 \
      --cap 2000000 --ladder-games 0 \
      --dump-buffer runs/pre_cuda_random/buffer.npz --out runs/pre_cuda_random

Format: the frozen compact rows (ROW_BYTES=223, version 1) + solve offsets,
expanded per batch by the Rust encoder. Network: flat encoder, h384/dg64/r64,
holding residual + id embedding, head == hidden. Generation runs with the
solver node cap (200k nodes; capped decisions fall back to uniform).

## What we learned

- 161 ReBeL epochs in 270 minutes (~100 s/epoch at 48 games; gen dominates).
  Generation is far slower than the starter-draft runs (~18 s/epoch) because
  random drafts with the Warrior Priest pair have 2-3x the config space and
  a fat tree-size tail; the solver node cap (200k) keeps the worst roots
  bounded. ~4.5M rows generated; the 2M-row buffer turned over completely.
- The dump is the deliverable: 2,000,000 rows, 54,624,298 configs, 254,270
  solves, format version 1, rules hash verified. Rows are the frozen
  223-byte format with solve offsets, so the section-7 comparisons can
  split at solve boundaries.
- Per-epoch loss settled ~0.010; age buckets stayed close; horizon games
  fell to ~0% after the cap payoff annealed; cfgs/decision grew 7 -> 16 over
  the run as the net tightened.
- 38 snapshots; the final is snap_37.pt. Nothing was laddered (by design:
  this run's Elo is not a decision metric).
- The 1,000-root GPU-sizing sample (roots.bin, sampled with snap_37 pushed)
  gives the depth-2/3/4 tree-size table (all-zero net, 2M-node cap):

```
depth 2 (0/1000 capped):        med      p95      p99
  nodes                         661    4,522   18,072
  leaves                        427    3,101   13,810
  action cells                5,166   71,773  158,800
  configs                     9,711   94,382  293,569
  upload MB                     1.7     12.1     52.0
  build ms                      0.9      7.4     26.4

depth 3 (13/1000 capped):      med      p95      p99
  nodes                       9,091   87,919  298,665
  leaves                      8,123   76,912  254,477
  action cells              122,707 1,343,979 4,889,616
  configs                   126,729 1,517,213 3,901,795
  upload MB                    32.5    306.8  1,026.6
  build ms                     14.0    164.3  1,944.3

depth 4 (100/1000 capped):    med      p95      p99
  nodes                     204,747 1,392,413 1,819,302
  leaves                    124,658   933,205 1,481,913
  action cells            1,826,702 22,390,115 51,835,757
  configs                 2,918,943 22,277,864 44,953,472
  upload MB                   521.6  3,849.9  5,791.8
  build ms                   425.8  3,343.3  4,940.7
```

The live GPU pool is sized from p99 of the depth the deployment uses; the
cap-hit rate at depth 4 (10%) says the GPU build needs the same cap, passed
as metadata.

## State of the project at this point

- Pre-CUDA engineering complete: WP + MainPlay-only queries, frozen row
  format, network wiring (residual, head width, id embedding), hex-neighbour
  candidate (Python), offline harness (solve-aligned, seeds, lr sweep),
  roots + treesize tooling, docs/TREE.md contract v1.
- Gates: 75 Rust tests, parity (incl. head split), format E2E test,
  10-minute random-draft smoke.
- Old 8-dim checkpoints retired with the format freeze; the v1 path still
  loads the pre-describer pool until it rotates.
