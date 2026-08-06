# arch_policy — new architecture with policy head (65 min)

## What we were trying
Same fresh baseline as arch_base, with the policy head trained (--policy 0.3;
per-solve belief-weighted cross-entropy on the solve's reference strategy) — the
only change. One-change gate against arch_base at equal wall-clock.

## Config
--minutes 65 --warm-minutes 5 --snapshot-every 20 --depth 2 --iters 64 --seed 21
--hidden 384 --dg 64 --rank 64 --de 32 --cap 2000000 --train-gen-ratio 4
--warm-games 96 --rebel-games 48 --aux 0 --ladder-games 0
(other cfg from log.json: lr 0.001, decayed at 0.33/0.67 of the run, batch 1024,
cfgs_per_row 48, explore 0.25, temp 2.0, eval_mix 0.5, cap_value 0.04,
anneal_frac 0.4, mc_mix 0, recent_mix 0.5, recent_frac 0.2, random_draft off)
plus --policy 0.3

## Snapshots (log.json)
init t=306s snap_00.pt | s1 t=1511s snap_01.pt | s2 t=2729s snap_02.pt | final t=3901s snap_03.pt

## Results
Ladder of all 12 checkpoints of the three arch runs (runs/ladder_arch/NOTES.md,
40 games/pairing, iters=64, greedy=0): final 798±19, s2 748±19, s1 663±20,
init 175±28. Within ±20 of arch_base at every rung.

Ladder of the three finals vs the two old 1-hour snapshots
(runs/ladder_arch_finals/NOTES.md, 100 games/pairing): arch_policy.final 646±18,
the lowest of the three arch finals. Head-to-head (W/L/D): vs arch_base.final
47/50/3, vs arch_dcfr.final 43/56/1, vs t64_turbo_s14.s1 36/59/5,
vs t64_h384_dg64_s11.s1 27/69/4.

## State of the project
As arch_base. The policy head is also used by --warm (see arch_policy_warm,
killed; no results).
