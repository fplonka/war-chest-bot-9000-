# t256_h384_dg64_s12_r2 — rerun of the crashed t256 run, with the desync fix

**Date:** 2026-08-06/07

## What was run

An exact rerun of `t256_h384_dg64_s12` (seed 12, 270 min, warm 5 min, iters
256, hidden 384, dg 64, snapshots every 60 min, no ladder), from commit
`d70451b` — the original engine plus the one change that killed the original:
`Belief::from_pairs` no longer drops a config whose weight underflowed to
exactly 0.0 (support is reachability, never weight). Output
`runs/t256_h384_dg64_s12_r2`.

## What happened

Ran the full 270 minutes, 0 panics. The original died at epoch 168 (~89 min);
the rerun passed that point (and the same seed's games) without incident, and
finished at ~277 epochs.

## What was measured

The three-run merged ladder (`runs/ladder_3runs`, all 18 snapshots of the
three 270-min runs + Greedy + Random, 40 paired games per pairing, rated at
depth 2 / 64 iters — the gate convention):

| snapshot | t64 ctrl (s11) | t64 wide (s13) | t256 (s12_r2) |
|---|---|---|---|
| init  | 317 | 271 | 310 |
| s1    | 876 | 810 | 699 |
| s2    | 927 | 889 | 756 |
| s3    | 1008 | 924 | 830 |
| s4    | 1021 | 968 | 831 |
| final | 1033 | 970 | 853 |

Greedy 131, Random 0. Standard errors ±15-16.

Verdicts, per the plan's step-1 decision rules:

- **t256 is clearly below ctrl at every matched time point** (853 vs 1033
  final). Solve quality (more CFR iterations per solve) is not the binding
  constraint at these data rates — t256 trades ~4x the epoch throughput for
  better-converged targets, and the trade loses.
- **wide is below ctrl at every point** (970 vs 1033 final). Capacity is not
  the constraint either.
- **ctrl is still climbing at 270 min** (s3 1008 -> final 1033). Long runs pay;
  the plateau elo01 showed at 30 min is slow convergence, not a wall.

Together: none of the three knobs (time, iters, width) is the constraint by
itself. The plan's fallback — the depth probe (step 6) — and the under-training
hypothesis both move up.

## State of the project at this point

- turbo merged into master with the remaining pieces (log-spaced snapshot
  thinning, solve counts for the trainer ratio, depth-counting fix, per-side
  eval settings, age-bucket loss, pool file); pushed as 6508d52..abea357.
- Pool: `runs/pool.json` = t64_h384_dg64_s11.final (1033), s4 (1021).
- Next: 3a (turbo T=64, d=2, 4.5h), then 3b (turbo T=512, d=2, 4.5h), then
  the turbo ladders and the finals-only ladder against runs 1-3.

Raw results — ladder_3runs (all 18 snapshots of the three 270-min runs + refs, 40 games/pairing at depth 2 / 64 iters):

```
=== Elo (runs/ladder_3runs, random = 0) ===
                      player   trained     elo    +-   score
     t64_h384_dg64_s11.final  270.0min    1033    16   0.777
        t64_h384_dg64_s11.s4  245.5min    1021    16   0.766
        t64_h384_dg64_s11.s3  185.3min    1008    16   0.752
     t64_h512_dg96_s13.final  270.2min     970    15   0.713
        t64_h512_dg96_s13.s4  245.9min     968    15   0.711
        t64_h384_dg64_s11.s2  125.2min     927    15   0.666
        t64_h512_dg96_s13.s3  185.8min     924    15   0.663
        t64_h512_dg96_s13.s2  125.7min     889    15   0.624
        t64_h384_dg64_s11.s1   65.0min     876    15   0.609
 t256_h384_dg64_s12_r2.final  270.2min     853    15   0.583
    t256_h384_dg64_s12_r2.s4  248.1min     831    15   0.559
    t256_h384_dg64_s12_r2.s3  187.3min     830    15   0.558
        t64_h512_dg96_s13.s1   65.4min     810    15   0.536
    t256_h384_dg64_s12_r2.s2  126.8min     756    16   0.479
    t256_h384_dg64_s12_r2.s1   65.9min     699    16   0.422
      t64_h384_dg64_s11.init    5.0min     317    23   0.164
  t256_h384_dg64_s12_r2.init    5.0min     310    24   0.161
      t64_h512_dg96_s13.init    5.0min     271    24   0.142
                      greedy         -     131    27   0.080
                      random         -       0    33   0.035
```
