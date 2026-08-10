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
- [ ] **Recover the ~20% throughput lost since `gpu_golden8`.** 1,066 solves/s
      against 1,315 on the same hardware, measured with nothing else running.
      `tgt_std` tracks solves rather than wall clock, so this is 20% of the
      progress of every run. Bisect 22f63f6 and 0192e4a.
- [ ] **Make the value network antisymmetric.** `v_0 + v_1` is +0.050 mean,
      0.045 std, against a 0.354 value spread — 180x the target bias from
      stopping CFR at T=64, and it is the network rather than the solve. See
      `runs/solvererr_g8/NOTES.md`.
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
