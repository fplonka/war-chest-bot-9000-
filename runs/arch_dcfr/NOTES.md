# arch_dcfr — new architecture with DCFR solver (65 min)

## What we were trying
Same fresh baseline as arch_base, with the solver switched from linear CFR to
DCFR (--cfr dcfr; α=1.5, β=0, γ=2, the TurboReBeL setting) — the only change.
One-change gate against arch_base at equal wall-clock.

## Config
--minutes 65 --warm-minutes 5 --snapshot-every 20 --depth 2 --iters 64 --seed 21
--hidden 384 --dg 64 --rank 64 --de 32 --cap 2000000 --train-gen-ratio 4
--warm-games 96 --rebel-games 48 --aux 0 --ladder-games 0
(other cfg from log.json: lr 0.001, decayed at 0.33/0.67 of the run, batch 1024,
cfgs_per_row 48, explore 0.25, temp 2.0, eval_mix 0.5, cap_value 0.04,
anneal_frac 0.4, mc_mix 0, recent_mix 0.5, recent_frac 0.2, random_draft off)
plus --cfr dcfr

## Snapshots (log.json)
init t=300s snap_00.pt | s1 t=1513s snap_01.pt | s2 t=2730s snap_02.pt | final t=3922s snap_03.pt

## Results
Ladder of all 12 checkpoints of the three arch runs (runs/ladder_arch/NOTES.md,
40 games/pairing, iters=64, greedy=0): s2 809±19, final 806±19, s1 694±19,
init 182±27. Best-checkpoint (s2) and final both within ±20 of arch_base.final
(817) and arch_policy.final (798).

Ladder of the three finals vs the two old 1-hour snapshots
(runs/ladder_arch_finals/NOTES.md, 100 games/pairing): arch_dcfr.final 696±17,
the highest of the three arch finals. Head-to-head (W/L/D): vs arch_base.final
56/42/2, vs arch_policy.final 56/43/1, vs t64_turbo_s14.s1 43/56/1,
vs t64_h384_dg64_s11.s1 36/62/2.

## State of the project
As arch_base; --cfr is now wired through train.py argparse (commit 986acab).
