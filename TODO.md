# TODO

## Pre-CUDA (see /tmp/warchest-pre-cuda-plan.md)

- [ ] Record the section-7 decisions once runs/pre_cuda_random finishes:
      7a card/holding, 7b flat-vs-hex, 7c head width (train/pre_cuda_comparisons.sh).
- [ ] Write runs/pre_cuda_random/NOTES.md.
- [ ] Save the 1,000-root GPU-sizing sample from the trained net
      (python train/save_roots.py --ckpt runs/pre_cuda_random/<final>.pt ...)
      and fill the depth-2/3/4 tree-size table (examples/treesize).
- [ ] Delete the v1 compatibility path (engine/src/v1.rs, value_net_v1.py,
      from_flat_v1, PUBFEAT_V1) once the ladder pool has rotated past the
      old checkpoints; update runs/pool.json with post-freeze snapshots.
- [ ] The CPU solver's node cap is 200k in generation; the GPU pool sizing
      (docs/TREE.md) must budget the same tail, and the GPU build should
      receive the cap as metadata.

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
- [x] **The value target estimator was checked and is right.** The claim that
      it should be the per-iteration mean inside the CFR loop compares us to
      vanilla ReBeL. We implement TurboReBeL, whose Phase 2 (paper Algorithm 2,
      line 13) specifies `v^σ(β_{s,t}) ← UpdateCFV(S′, σ)` -- backpropagate
      under the *fixed final* reference strategy, for each intermediate PBS.
      That is exactly `value_under`, and Phase 2 is what earns the T+1 rows
      per decision that `selfplay_walk.rs` pins.
- [x] **The zero-reach uniform fallback does not fire.** Measured over a
      depth-2/64-iteration solve under a flat network -- the warm-start
      condition it was suspected to spoil -- across 114,585 reach rows and
      64,822 strategy rows: no reach exactly zero, none below 1e-30, no
      strategy sum zero. The 1e-80 floor the reference uses is not even
      representable in the f32 arena. Left alone deliberately.
