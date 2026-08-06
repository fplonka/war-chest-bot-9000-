# t512_turbo_s15 — turbo at T=512, depth 2, 4.5 hours (gate 3b)

**Date:** 2026-08-07

## What was run

The payoff test for turbo: same recipe as 3a (`t64_turbo_s14`) at iters 512
instead of 64 — 270 min, warm 5 min, snapshots every 60 min, seed 15,
depth 2, hidden 384, dg 64, cap 2M. Without turbo this would have cost ~8x
the epoch time *and* ~8x less data per epoch; with turbo the data rate is
flat in T (11 rows per solve at T=512 vs 8 at T=64, thanks to the
log-spaced thinning) and the only cost is the solve itself.

## What happened

Full 270 minutes, 191 epochs (~2.6 min/epoch), 6 snapshots, 0 panics, no
crash. Rows per epoch ~22-23k, the same as 3a's ~18-22k — the data rate
really is roughly flat in T; the epoch time is 8x (the CFR iterations
themselves).

## What was measured

Not yet laddered at the time of writing (the turbo-vs-turbo ladder and the
finals ladder run right after). Notes from the run's own log:

- Per-epoch loss ~0.013, similar to 3a's ~0.011-0.013. Training is doing
  about the same amount of work per epoch (per-solve step count) on
  better-converged targets (solvererr at T=512 is ~0.0001 vs 0.00036 at
  T=64).
- Only 191 epochs in 4.5h — the network saw ~4.3M rows, roughly a quarter
  of 3a's 17M, and the 2M buffer never fully turned over (~44% fill at
  the end). If T=512 wins the ladder it will be on target quality, not
  data volume.
- Age buckets stayed close together, as in 3a.

## State of the project at this point

- master has moved on (policy-head / action-features / CFR-variants work
  landed after these runs); the ladders ran from a worktree at 4aa86ca,
  the commit the runs actually used.
- Ladders done: turbo-vs-turbo (runs/ladder_turbo) and finals-only
  (runs/ladder_finals). Pool: s11.final (919), t64_turbo_s14.final (904).

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
