# ladder_arch_finals — the three arch finals vs the old 1-hour snapshots

Run 2026-08-06, ~22:03–22:19. 6 players, 15 pairings, 100 paired games each.

Players: greedy (pinned at 0), the three arch finals, and the 65-minute (s1)
snapshots of the two old 4.5-hour runs, entered via a pool file:
  t64_turbo_s14.s1        snap_01.pt, t=3922s (turbo T=64, 4.5h run)
  t64_h384_dg64_s11.s1    snap_01.pt, t=3901s (non-turbo T=64, 4.5h run)
The old runs' `final` snapshots are 4.5h checkpoints, so the 65-minute s1
snapshots are the equal-training-time comparison (same convention as
runs/ladder_s1 and runs/ladder_s1_t256). An earlier attempt at this ladder used
the old runs' `final` snapshots and was discarded at 14/15 pairings; this
run.log/ladder.json are the corrected run only.
Settings: depth=2, iters=64, temp=2.0, seed=7, refs=greedy only.
Elo: as in ladder_arch (Zermelo MM, one prior draw per pairing, greedy = 0).

## Elo
          t64_h384_dg64_s11.s1  65.0min   773.4 ±17.9  score 0.699
              t64_turbo_s14.s1  65.4min   733.0 ±17.5  score 0.642
               arch_dcfr.final  65.4min   695.6 ±17.4  score 0.588
               arch_base.final  65.1min   663.1 ±17.5  score 0.541
             arch_policy.final  65.0min   646.3 ±17.6  score 0.517
                        greedy        -     0.0 ±58.8  score 0.013

## Pairings (W = first-named player's wins)
                        greedy vs arch_base.final            W  0 L 95 D  5 score 0.025
                        greedy vs arch_dcfr.final            W  0 L100 D  0 score 0.000
                        greedy vs arch_policy.final          W  0 L 98 D  2 score 0.010
                        greedy vs t64_turbo_s14.s1           W  0 L 97 D  3 score 0.015
                        greedy vs t64_h384_dg64_s11.s1       W  1 L 98 D  1 score 0.015
               arch_base.final vs arch_dcfr.final            W 42 L 56 D  2 score 0.430
               arch_base.final vs arch_policy.final          W 50 L 47 D  3 score 0.515
               arch_base.final vs t64_turbo_s14.s1           W 34 L 61 D  5 score 0.365
               arch_base.final vs t64_h384_dg64_s11.s1       W 41 L 57 D  2 score 0.420
               arch_dcfr.final vs arch_policy.final          W 56 L 43 D  1 score 0.565
               arch_dcfr.final vs t64_turbo_s14.s1           W 43 L 56 D  1 score 0.435
               arch_dcfr.final vs t64_h384_dg64_s11.s1       W 36 L 62 D  2 score 0.370
             arch_policy.final vs t64_turbo_s14.s1           W 36 L 59 D  5 score 0.385
             arch_policy.final vs t64_h384_dg64_s11.s1       W 27 L 69 D  4 score 0.290
              t64_turbo_s14.s1 vs t64_h384_dg64_s11.s1       W 39 L 57 D  4 score 0.410
