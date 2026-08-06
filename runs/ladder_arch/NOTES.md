# ladder_arch — all checkpoints of the three arch runs, round robin

Run 2026-08-06, ~21:26–21:55. 13 players, 78 pairings, 40 paired games each.

Players: greedy (pinned at 0) + all 12 snapshots of arch_base, arch_dcfr and
arch_policy (labels init/s1/s2/final; snapshot times in each run's NOTES.md).
Settings: depth=2, iters=64, temp=2.0, seed=7, refs=greedy only.
Elo: Bradley-Terry fit by Zermelo MM iteration; one prior draw added per played
pairing; first reference pinned at 0; standard errors from the Fisher
information diagonal (read as "how well placed this player is", not as a
pairwise test). ladder.json holds the machine-readable copy.

## Elo
               arch_base.final  65.1min   817.3 ±19.5  score 0.723
                  arch_dcfr.s2  45.5min   808.8 ±19.4  score 0.714
               arch_dcfr.final  65.4min   806.0 ±19.4  score 0.710
                  arch_base.s2  45.5min   803.1 ±19.4  score 0.707
             arch_policy.final  65.0min   797.5 ±19.3  score 0.701
                arch_policy.s2  45.5min   747.8 ±19.1  score 0.645
                  arch_base.s1  25.2min   716.6 ±19.2  score 0.609
                  arch_dcfr.s1  25.2min   693.6 ±19.3  score 0.583
                arch_policy.s1  25.2min   662.6 ±19.5  score 0.549
                arch_base.init   5.0min   206.6 ±27.1  score 0.177
                arch_dcfr.init   5.0min   181.5 ±27.4  score 0.161
              arch_policy.init   5.1min   174.7 ±27.6  score 0.157
                        greedy        -     0.0 ±33.2  score 0.062

## Pairings (W = first-named player's wins)
                        greedy vs arch_base.init             W  3 L 24 D 13 score 0.237
                        greedy vs arch_base.s1               W  0 L 40 D  0 score 0.000
                        greedy vs arch_base.s2               W  0 L 39 D  1 score 0.013
                        greedy vs arch_base.final            W  0 L 40 D  0 score 0.000
                        greedy vs arch_dcfr.init             W  8 L 25 D  7 score 0.287
                        greedy vs arch_dcfr.s1               W  0 L 40 D  0 score 0.000
                        greedy vs arch_dcfr.s2               W  0 L 40 D  0 score 0.000
                        greedy vs arch_dcfr.final            W  0 L 39 D  1 score 0.013
                        greedy vs arch_policy.init           W  3 L 28 D  9 score 0.188
                        greedy vs arch_policy.s1             W  0 L 39 D  1 score 0.013
                        greedy vs arch_policy.s2             W  0 L 40 D  0 score 0.000
                        greedy vs arch_policy.final          W  0 L 40 D  0 score 0.000
                arch_base.init vs arch_base.s1               W  0 L 40 D  0 score 0.000
                arch_base.init vs arch_base.s2               W  0 L 38 D  2 score 0.025
                arch_base.init vs arch_base.final            W  0 L 39 D  1 score 0.013
                arch_base.init vs arch_dcfr.init             W 17 L 16 D  7 score 0.512
                arch_base.init vs arch_dcfr.s1               W  3 L 37 D  0 score 0.075
                arch_base.init vs arch_dcfr.s2               W  0 L 40 D  0 score 0.000
                arch_base.init vs arch_dcfr.final            W  0 L 39 D  1 score 0.013
                arch_base.init vs arch_policy.init           W 21 L 13 D  6 score 0.600
                arch_base.init vs arch_policy.s1             W  2 L 34 D  4 score 0.100
                arch_base.init vs arch_policy.s2             W  0 L 40 D  0 score 0.000
                arch_base.init vs arch_policy.final          W  0 L 38 D  2 score 0.025
                  arch_base.s1 vs arch_base.s2               W 14 L 26 D  0 score 0.350
                  arch_base.s1 vs arch_base.final            W 17 L 21 D  2 score 0.450
                  arch_base.s1 vs arch_dcfr.init             W 38 L  2 D  0 score 0.950
                  arch_base.s1 vs arch_dcfr.s1               W 25 L 14 D  1 score 0.637
                  arch_base.s1 vs arch_dcfr.s2               W 14 L 25 D  1 score 0.362
                  arch_base.s1 vs arch_dcfr.final            W 12 L 28 D  0 score 0.300
                  arch_base.s1 vs arch_policy.init           W 38 L  1 D  1 score 0.963
                  arch_base.s1 vs arch_policy.s1             W 17 L 23 D  0 score 0.425
                  arch_base.s1 vs arch_policy.s2             W 19 L 21 D  0 score 0.475
                  arch_base.s1 vs arch_policy.final          W 16 L 24 D  0 score 0.400
                  arch_base.s2 vs arch_base.final            W 17 L 22 D  1 score 0.438
                  arch_base.s2 vs arch_dcfr.init             W 40 L  0 D  0 score 1.000
                  arch_base.s2 vs arch_dcfr.s1               W 26 L 14 D  0 score 0.650
                  arch_base.s2 vs arch_dcfr.s2               W 16 L 21 D  3 score 0.438
                  arch_base.s2 vs arch_dcfr.final            W 23 L 17 D  0 score 0.575
                  arch_base.s2 vs arch_policy.init           W 39 L  1 D  0 score 0.975
                  arch_base.s2 vs arch_policy.s1             W 30 L  9 D  1 score 0.762
                  arch_base.s2 vs arch_policy.s2             W 22 L 18 D  0 score 0.550
                  arch_base.s2 vs arch_policy.final          W 19 L 20 D  1 score 0.487
               arch_base.final vs arch_dcfr.init             W 40 L  0 D  0 score 1.000
               arch_base.final vs arch_dcfr.s1               W 23 L 15 D  2 score 0.600
               arch_base.final vs arch_dcfr.s2               W 25 L 14 D  1 score 0.637
               arch_base.final vs arch_dcfr.final            W 16 L 24 D  0 score 0.400
               arch_base.final vs arch_policy.init           W 40 L  0 D  0 score 1.000
               arch_base.final vs arch_policy.s1             W 27 L 12 D  1 score 0.688
               arch_base.final vs arch_policy.s2             W 25 L 14 D  1 score 0.637
               arch_base.final vs arch_policy.final          W 23 L 14 D  3 score 0.613
                arch_dcfr.init vs arch_dcfr.s1               W  1 L 39 D  0 score 0.025
                arch_dcfr.init vs arch_dcfr.s2               W  1 L 39 D  0 score 0.025
                arch_dcfr.init vs arch_dcfr.final            W  0 L 40 D  0 score 0.000
                arch_dcfr.init vs arch_policy.init           W 16 L 16 D  8 score 0.500
                arch_dcfr.init vs arch_policy.s1             W  2 L 37 D  1 score 0.062
                arch_dcfr.init vs arch_policy.s2             W  0 L 39 D  1 score 0.013
                arch_dcfr.init vs arch_policy.final          W  1 L 36 D  3 score 0.062
                  arch_dcfr.s1 vs arch_dcfr.s2               W 18 L 22 D  0 score 0.450
                  arch_dcfr.s1 vs arch_dcfr.final            W 12 L 27 D  1 score 0.312
                  arch_dcfr.s1 vs arch_policy.init           W 38 L  2 D  0 score 0.950
                  arch_dcfr.s1 vs arch_policy.s1             W 23 L 17 D  0 score 0.575
                  arch_dcfr.s1 vs arch_policy.s2             W 14 L 22 D  4 score 0.400
                  arch_dcfr.s1 vs arch_policy.final          W 12 L 28 D  0 score 0.300
                  arch_dcfr.s2 vs arch_dcfr.final            W 22 L 16 D  2 score 0.575
                  arch_dcfr.s2 vs arch_policy.init           W 40 L  0 D  0 score 1.000
                  arch_dcfr.s2 vs arch_policy.s1             W 28 L 11 D  1 score 0.713
                  arch_dcfr.s2 vs arch_policy.s2             W 25 L 15 D  0 score 0.625
                  arch_dcfr.s2 vs arch_policy.final          W 22 L 17 D  1 score 0.562
               arch_dcfr.final vs arch_policy.init           W 40 L  0 D  0 score 1.000
               arch_dcfr.final vs arch_policy.s1             W 29 L 11 D  0 score 0.725
               arch_dcfr.final vs arch_policy.s2             W 23 L 16 D  1 score 0.588
               arch_dcfr.final vs arch_policy.final          W 16 L 24 D  0 score 0.400
              arch_policy.init vs arch_policy.s1             W  1 L 37 D  2 score 0.050
              arch_policy.init vs arch_policy.s2             W  0 L 40 D  0 score 0.000
              arch_policy.init vs arch_policy.final          W  0 L 39 D  1 score 0.013
                arch_policy.s1 vs arch_policy.s2             W 13 L 25 D  2 score 0.350
                arch_policy.s1 vs arch_policy.final          W 14 L 26 D  0 score 0.350
                arch_policy.s2 vs arch_policy.final          W 15 L 25 D  0 score 0.375
