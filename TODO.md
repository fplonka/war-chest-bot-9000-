# TODO

## The ML iteration loop

- [ ] **Run `cadence` first.** Does a 15-minute run rank changes the way a
      60-minute one does? If it does, every experiment below costs a quarter of
      what it costs now. Nobody runs this because it does not feel like
      progress; it sets the cadence for everything after it.
- [x] ~~Build the first ground-truth set off the strongest checkpoint we have.~~
      Done and the answer was negative: a set built that way ranks its own
      builder's neighbourhood above stronger networks (numbers in
      `train/truth.py`). It is not the ruler.
- [ ] **Build a terminal-anchored ground-truth set instead.** Late-game
      positions solved deep enough that every leaf of the subgame is terminal
      have exact game values with no evaluator in them, so the ruler stays true
      no matter how strong the networks get. `Conv::zero_sum` reaching zero is
      the check that a solve got there.
- [ ] **Make ladders cheap.** Three things found while running one, in
      decreasing certainty:
      *SPRT ran on one pairing only* — every other pairing played its full fixed
      count, so `greedy` vs a trained net spent 600 games proving a 0-91 result.
      Now applies everywhere (`ae5e4c1`), and those pairings stop at 100.
      *The tuned wave settings were not defaults* — only `train.py` set them, so
      a ladder ran 48k-row waves on an 800us timer and left both cards near half
      idle at 50-60% (`80009d9`). Hypothesis, not yet measured: rerun the same
      ladder on the new build and compare wall clock. 75ms waves could equally
      hurt a workload with less in flight than training has.
      *Evaluation could use fewer CFR iterations* — `--eval-search dcfr:16`
      ranks every checkpoint under one cheap search. Whether the ranking
      survives is being measured; whether it is faster depends on the same
      question as below, since a solve may not be iteration-bound at all.
- [ ] **Understand the long-run throughput fall.** Not a code regression: the
      golden8-era commit `0192e4a`, rebuilt and rerun on the clean box, shows
      the same curve as HEAD. An earlier entry here claimed a ~25% regression;
      that was wrong, and it came from comparing golden8's 25 minutes against
      `base4h`'s 4-hour average. At equal age they agree (ReBeL minutes 15-25:
      golden8 ~1200-1380, base4h 1022).

      The curve is a dip and a recovery, then a slow fall over hours:
      minute 0-1 runs unthrottled (~2300/s) because the trainer has no debt
      yet; minutes 1-4 collapse (203/s) because almost no games have *finished*
      -- `games` per epoch is 0 or 1 and nothing has recycled; minutes 4-8
      recover to 1331/s as games complete (11 -> 42 -> 102 -> 192). Only after
      ~25 minutes does the slow fall start, which is why golden8 never saw it.

      The lead is tree size, not belief support. Configs per decision *fall*
      (29.6 -> 15.9) as reserves drain, but `node_caps` -- subgames hitting the
      200k node ceiling -- rises from 75 to 236 per epoch over `base4h` while
      golden8 stays in 14-152 for its whole life. Better play means longer games
      and more midgame positions with many legal actions. `gpu_wait_s` falling
      from 239 to ~50 agrees: the workers wait less on the card, so the limit
      moved onto the host, which is where trees are built.

      Next: instrument nodes per solve directly (the epoch record has
      `node_caps` and `oversize_routes` but no node count), and check whether
      the node cap is being hit often enough to distort the search as well as
      slow it.

- [ ] **Make the value network antisymmetric.** `v_0 + v_1` is +0.025 mean and
      0.032 absolute against a 0.416 value spread — 8% of the signal, a third of
      the network's own error, and ~130x the target bias from stopping CFR at
      T=64. It is the network rather than the solve, and a random network is off
      by the same amount. `train.py::zero_sum` projects it out of the targets;
      whether that buys strength is untested. See `runs/solvererr_g8/NOTES.md`.
- [ ] **Does the centred seat bit remove the violation in a *trained* net?**
      `35f37ee` centres the seat channel, which removes the common-mode offset
      at initialisation: mean +0.0398 -> +0.0008 over 40 seeds, and the sign
      stops being 85% predictable. That is measured at initialisation only.
      Re-measure `v_0 + v_1` on a checkpoint trained with the new build. The
      mean should collapse. The state-dependent part (sd ~0.032) should not,
      because nothing constrains it — and that is the part that changes play.
      A projection that forced the sum to zero was written and then removed
      (`train.py::zero_sum`, in the history at `8afa4b4`). It was untested, it
      corrected a symptom rather than a cause, and it would have destroyed the
      instrument: with the projection on, `v_0 + v_1` is zero by construction and
      can no longer tell us anything. Bring it back only if the trained
      measurement shows the state-dependent part still costs strength. If it
      does, enforce it in `Solver::readout`, where the search actually reads the
      leaves, rather than in the labels.
- [ ] Re-run the `dcfr` / `aux` / `policy` experiments through `exp.py`. The
      2026-08-06 `arch_*` results are **not** evidence: three arms inside
      ±20 Elo at 100 games per pairing, which resolves nothing finer than ~70.

## Measurements waiting on the machine

- [ ] **The regret rule.** `examples/solvererr.rs` on a trained net. DCFR at
      T=64 reaches what linear needs T=512 for; free accuracy at the T we
      run. The dump run's solvererr gate can reuse its targets.
- [ ] **A4 / warm start.** Compare warm-start methods with `solvererr` once a
      strong policy head exists (the pre-CUDA plan leaves warm off; the GPU
      API only needs optional initial regret/strategy arrays).
- [ ] **A2 offline.** The card/holding screen (7a) on the random-draft dump
      is exactly this, now that the dump exists to run it on.

## Known gaps

- [ ] **`gpu::tests::wave_composition_stays_bounded` fails.** A tree's strategy
      moves by 1.54x the allowed tolerance depending on which other jobs share
      its wave (`gpu 5.355e-1` against `reference 6.924e-1`). The GPU test
      module did not compile until `keep_states` was added to its `TEST_CFG`,
      so this has been unmeasured rather than passing. `full_wave_oracle` and
      `zero_network_oracle` do pass. Probably reduction order in the batched
      GEMMs, but that is a guess and it should be a measurement.

- [ ] **The solver's own tests do not compile.** `tests/rebel_solver.rs`,
      `tests/rebel_pbs.rs` and `examples/wave_tape.rs` reference `TNode::s`,
      which the GPU tree-builder work removed. They have been dark since that
      refactor: 20 compile errors at b55c631, none of them new. The PBS and
      solver oracles are the correctness floor under everything the agent
      does, so this is the first thing to fix.

- [ ] **The horizon manufactures draws.** A game is cut at 256 coin plays and
      scored as a draw; War Chest has no draws. Report the draw rate beside
      every score; raise MAX_MAIN_PLAYS and check the cap rate; treat Greedy
      as a draw generator. Scoring a timeout as -1 for both was rejected:
      it makes the game non-zero-sum, and zero-sum is load-bearing.
- [ ] **The subgame is not quite zero-sum.** Its leaves are network values;
      predicting v_0 and defining v_1 = -v_0 would enforce it by construction.
- [x] Draw transitions: round-start runs are now one direct multivariate
      hypergeometric per source config (see rebel.rs::DrawScratch::run),
      pinned to the old composition by test.
- [ ] Validate the mirror augmentation against the engine rather than against
      invariants (a State::mirror() in Rust would make the encoder the oracle).
