# TODO

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
- [ ] The horizon payoff (`CAP_MARKER_VALUE`) is a rule we invented. It is
      annealed to zero over the first 40% of the ReBeL phase and evaluation
      always runs at zero, so the shipped checkpoint is fitted to the real game
      — but confirm the cap-hit rate really reaches ~0 on a long run rather than
      assuming it, and consider whether the anneal can start lower.
- [ ] The belief reaches the network as a fixed-width sum of learned config
      embeddings (`dg`). That is the one place a fixed width is a real
      approximation: a belief is a distribution over a config space too large to
      enumerate, so unlike the value's dependence on a *single* config, this one
      cannot be made exact at fixed width. Sweep `--dg` offline and find out
      what it costs.
- [ ] The big run (9h, this machine). Settings the measurements point to:
      `--iters 16 --cap 2000000 --warm-minutes 5 --gate-every 1200 --gate-vs both`.
      Rationale for each is in `docs/REBEL.md` sections 5 and 7.
- [ ] Random-draft training runs. The encoding now carries each card's tactic
      and attribute flags, so a draft the network has never seen is describable
      rather than an unseen identity code -- which was the prerequisite.
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
