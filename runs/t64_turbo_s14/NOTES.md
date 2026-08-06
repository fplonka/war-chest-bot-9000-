# t64_turbo_s14 — turbo at T=64, depth 2, 4.5 hours (gate 3a)

**Date:** 2026-08-07

## What was run

The first turbo run at the overnight length, so it is directly comparable to
run 1 (`t64_h384_dg64_s11`): 270 min, warm 5 min, snapshots every 60 min,
seed 14, depth 2, iters 64. Everything else identical to run 1's recipe
(hidden 384, dg 64, cap 2M, train_gen_ratio 4). Differences from run 1, all
shipped together: turbo generation (T+1 rows per solve, thinned to the ~8
log-spaced iterates + the live belief), training steps counted per solve,
depth counted in completed coin plays (tactic micro-choices ride free), the
belief-underflow fix, and the age-bucket loss in the log.

## What happened

Full 270 minutes, 819 epochs (~20 s/epoch), 6 snapshots, 0 panics, no crash.
Rows per epoch ~18-22k (vs ~4k for run 1): the buffer turns over ~5x faster.

## What was measured

Two ladders, both at the gate's common settings (depth 2, 64 iters, 40
paired games per pairing, run from a worktree at the training commit):

- Turbo-vs-turbo (`runs/ladder_turbo`): **t64_turbo_s14 dominates
  t512_turbo_s15 at every snapshot** — final 1099 vs 767, s4 1051 vs 740.
  The T=512 run's better-converged targets did not make up for its data
  volume (191 epochs vs 819; the 2M buffer never fully turned over).
- Finals only (`runs/ladder_finals`, run1/run2/run3 finals + both turbo
  finals + refs): ctrl final 919, **t64_turbo_s14.final 904**, wide final
  841, t256 rerun final 793, t512_turbo_s15.final 634, greedy 226,
  random 0. Standard errors ±28-31.

Verdict for gate 3a: **no detected change** (904 vs 919, within the ±25
band). Turbo at T=64 is not visibly better than the plain run at the same
settings — but note this bundles the depth-counting fix and per-solve step
counting, so it is "the new base", not "turbo alone". The interesting
number is 3b's: turbo made T=512 affordable, and T=512 lost anyway — data
volume beats target quality, consistent with the under-training story.

Per-epoch loss settled around 0.011-0.013, similar to run 1's 0.013, on
targets that are better-converged (the turbo target's solvererr is 0.00036
vs 0.0008 at T=64). Age buckets stayed close together all run
(old ≈ new ≈ total), i.e. no visible staleness gap. tgt_std climbed slowly
(0.53 -> 0.64), pstd 0.34 -> 0.42 — the value function keeps spreading,
no collapse.

## State of the project at this point

- master has moved on (policy-head / action-features / CFR-variants work
  landed after these runs); the ladders ran from a worktree at 4aa86ca,
  the commit the runs actually used.
- Pool: s11.final (919), t64_turbo_s14.final (904), per the finals ladder.

Raw results — ladder_turbo (turbo64 vs turbo512, all snapshots, 40 games/pairing at depth 2 / 64 iters):

```
=== Elo (runs/ladder_turbo, random = 0) ===
                      player   trained     elo    +-   score
         t64_turbo_s14.final  270.1min    1099    24   0.859
            t64_turbo_s14.s4  245.9min    1051    22   0.821
            t64_turbo_s14.s3  185.8min    1050    22   0.820
            t64_turbo_s14.s2  125.4min     984    21   0.764
            t64_turbo_s14.s1   65.4min     863    20   0.649
        t512_turbo_s15.final  270.0min     767    20   0.554
           t512_turbo_s15.s3  189.8min     755    20   0.542
           t512_turbo_s15.s4  251.0min     740    20   0.528
           t512_turbo_s15.s2  127.8min     693    20   0.482
           t512_turbo_s15.s1   65.0min     641    21   0.434
         t512_turbo_s15.init    5.0min     374    26   0.229
          t64_turbo_s14.init    5.0min     297    27   0.182
                      greedy         -     135    31   0.096
                      random         -       0    38   0.040
```

Raw results — ladder_finals (run1/2/3 + both turbo finals + refs, 40 games/pairing at depth 2 / 64 iters):

```
=== Elo (runs/ladder_finals, random = 0) ===
                      player   trained     elo    +-   score
     t64_h384_dg64_s11.final  270.0min     919    30   0.777
         t64_turbo_s14.final  270.1min     904    29   0.760
     t64_h512_dg96_s13.final  270.2min     841    28   0.688
 t256_h384_dg64_s12_r2.final  270.2min     793    28   0.631
        t512_turbo_s15.final  270.0min     634    31   0.456
                      greedy         -     226    47   0.154
                      random         -       0    58   0.033
```
