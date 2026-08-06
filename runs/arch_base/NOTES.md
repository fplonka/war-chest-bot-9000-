# arch_base — new-architecture baseline (65 min)

## What we were trying
The first training run on the post-describer architecture: the public encoding
describes each coin type by its 25 rulebook facts and refers to cards by
coin-type index (no unit identity), so an unseen draft is describable instead of
an unknown identity code. This run is the fresh baseline every other new-arch
run gates against — one change per run, nothing else on (policy 0, aux 0,
linear CFR). Equal wall-clock with the other arch runs.

## Config
--minutes 65 --warm-minutes 5 --snapshot-every 20 --depth 2 --iters 64 --seed 21
--hidden 384 --dg 64 --rank 64 --de 32 --cap 2000000 --train-gen-ratio 4
--warm-games 96 --rebel-games 48 --aux 0 --ladder-games 0
(other cfg from log.json: lr 0.001, decayed at 0.33/0.67 of the run, batch 1024,
cfgs_per_row 48, explore 0.25, temp 2.0, eval_mix 0.5, cap_value 0.04,
anneal_frac 0.4, mc_mix 0, recent_mix 0.5, recent_frac 0.2, random_draft off)

## Snapshots (log.json)
init t=300s snap_00.pt | s1 t=1511s snap_01.pt | s2 t=2728s snap_02.pt | final t=3903s snap_03.pt

## Results
Ladder of all 12 checkpoints of the three arch runs (runs/ladder_arch/NOTES.md,
40 games/pairing, iters=64, greedy=0): final 817±20, s2 803±19, s1 717±19,
init 207±27. Monotone rise with training time.

Ladder of the three finals vs the two old 1-hour snapshots
(runs/ladder_arch_finals/NOTES.md, 100 games/pairing): arch_base.final 663±18.
Head-to-head (W/L/D): vs arch_dcfr.final 42/56/2, vs arch_policy.final 50/47/3,
vs t64_turbo_s14.s1 34/61/5, vs t64_h384_dg64_s11.s1 41/57/2.

## State of the project
New architecture (card describer + policy-head weights) is on master, built and
installed. Three arch runs done; arch_policy_warm was killed at ~13.5 min
(planned) and the two ladders above ran instead. Old pool runs (4.5h) still the
strongest rated models; their s1 snapshots rate above all three arch finals.
