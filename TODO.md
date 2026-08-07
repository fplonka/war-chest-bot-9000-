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
- [ ] Draw transitions are the largest remaining non-network CPU cost. A run
      of k draws is composed step by step over supports that grow ~5x each
      time; the multivariate hypergeometric gives the same answer directly.
      The solver node cap (200k) bounds the damage until then.
- [ ] Validate the mirror augmentation against the engine rather than against
      invariants (a State::mirror() in Rust would make the encoder the oracle).
