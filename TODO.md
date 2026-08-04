# TODO

- [ ] turbo rebel, higher T
- [ ] isomorphism: 5 coin slots ordering, for non fixed draft

- [ ] **Put the Warrior Priest back in the game.** Units 18 and 54 are missing
      from `DRAFT_POOL` in `engine/src/selfplay.rs` — two of nineteen — because
      the attribute triggers a *private* mid-round draw, which adds "which coin
      must I now play" to the private state. That was a reason to widen the
      private state, not to delete the unit: the agent currently cannot play
      part of War Chest at all, which is the same kind of compromise as the
      hand-keyed value function and a larger one. The fix is
      `Config { hand, fd, pending_coin }` — at most 5x the config count for the
      span of one coin play, still exactly enumerable — plus removing the
      Warrior Priest clamp from the subgame's chance handling.
- [ ] **The horizon, and the draws it manufactures.** A game is cut off at 256
      coin plays and scored as a draw. War Chest has no draws, so every one is
      an outcome we invented: 29 of 400 evaluation games vs Greedy in
      `runs/cfgvalue01` (7.3%), 10 of 400 vs the initial checkpoint. The
      generation-time cap rate is only 0-4%, so reading that number alone
      understates it.
      Three things to do, in order.
      * Report the draw rate beside every score instead of folding draws into
        0.5. A score of 0.961 with 7% draws is a different claim from 0.961 with
        none.
      * Raise `MAX_MAIN_PLAYS`. It is a `u16` const and the cost falls only on
        games that would run long. 256 was calibrated against site data saying
        real games top out near 200 actions — but our games plainly run longer
        than human games, so it was calibrated on the wrong distribution. Then
        check the cap rate rather than assuming.
      * Treat Greedy as a draw generator, not a neutral yardstick. Measured:
        deterministic Greedy *self-play* times out **100%** of the time, with
        neither marker count moving for the last ~240 of 256 plays. The stalling
        is a property of the weak reference opponent. The Elo ladder mostly
        routes around this — the snapshots' ratings are set by games against each
        other, and Greedy and Random only anchor the scale — but the draws are
        still there in the pairings that involve them.
      Scoring a timeout as -1 for both players was considered and rejected: it
      makes the game non-zero-sum, and zero-sum is load-bearing here — ReBeL's
      guarantees, CFR's convergence to Nash, `tests/rebel_solver.rs`'s oracle
      (the two players' root values must sum to zero), and the antisymmetry that
      `blend_outcome` and the warm start rely on. Worth revisiting only if
      raising the horizon fails.
- [ ] The belief reaches the network as a fixed-width sum of learned config
      embeddings (`dg`). That is the one place a fixed width is a real
      approximation: a belief is a distribution over a config space too large to
      enumerate, so unlike the value's dependence on a *single* config, this one
      cannot be made exact at fixed width. Sweep `--dg` offline and find out
      what it costs.
- [ ] **T = 512.** `--iters 64` is the current default and was itself a step
      back from a throughput-driven 16 (`docs/REBEL.md` section 7). The paper
      runs 256-1024. The measurement to make is not a loss curve — changing T
      changes the target function — but two runs of equal wall-clock at
      different T, rated on one ladder against each other and against a common
      Greedy anchor. That comparison needs `ladder.py` to accept snapshots from
      more than one run directory; it currently takes one.
- [ ] The big run (9h, this machine). Settings the measurements point to:
      `--iters 64 --cap 2000000 --warm-minutes 5 --snapshot-every 20`.
      Rationale for each is in `docs/REBEL.md` sections 5 and 7.
- [ ] Random-draft training runs. The encoding now carries each card's tactic
      and attribute flags, so a draft the network has never seen is describable
      rather than an unseen identity code -- which was the prerequisite. The
      draft generator itself was dealing both sides off the pool independently
      and has been fixed; no run to date used it.
- [ ] Validate the mirror augmentation against the engine rather than against
      invariants. `train/mirror.py` derives its permutation from exported
      layout offsets and checks involution plus what must and must not move; a
      `State::mirror()` in Rust would let the encoder itself be the oracle.
- [ ] Draw transitions are the largest remaining non-network cost (13%). A run
      of k draws is composed step by step over supports that grow ~5x each
      time; the multivariate hypergeometric gives the same answer directly from
      the parent support, needing a fallback for the mid-run reshuffle and an
      oracle against the chain.
- [ ] Revisit capacity once data stops being the constraint. At present the
      network memorises (`docs/REBEL.md` section 5), so extra parameters buy
      nothing; `--hidden 512` was the best architecture tested once
      augmentation removed the overfitting, by 1.5%.

