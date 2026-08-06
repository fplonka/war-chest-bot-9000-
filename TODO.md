# TODO

- [ ] i don't reallyyyy like the "log T" number of rows we keep with turbo, i'd like experiments with keepign all of them or at least keeping a higher log base number, so T 64 -> 512 gets us more than 3 more rows per thing
- [ ] isomorphism: 5 coin slots ordering, for non fixed draft

## Measurements waiting on the machine

- [x] **D0, half of it.** T=512 lost ~270 Elo to T=64 at equal wall-clock even
      with turbo removing the data cost (`8121f3c`). Data volume beats target
      quality at this scale; T=64 stays the default. Two consequences: the GPU
      work package exists to pay for large T and no longer has a reason to,
      and every "reach a given solve quality in fewer iterations" argument is
      worth much less than it looked.
- [ ] **The regret rule.** `examples/solvererr.rs` on a trained net, ~9 min for
      100 positions. Preliminary on 10 positions: DCFR at T=64 reaches the
      NashConv linear needs T=512 for. Read against D0 this is no longer a way
      to lower T — T is already low — but it is free accuracy at the T we run,
      so it is still worth a confirming gate.
- [ ] **A4.** Whether the warm start pays. `solvererr <weights> <n> <depth>
      <play> <skip> <warm>` reports every rule cold and warm; adopt only if warm
      at T/2 beats cold at T. Needs a checkpoint whose policy head is actually
      trained. D0 lowers the prior on this: if solve quality is not the binding
      constraint, arriving at it sooner buys little.
- [ ] **A2 offline.** Fit the card-describer network against a fixed-draft one on
      a **random-draft** dump. It deletes unit identity, which is a perfect
      feature on the starter matchup, so a fixed-draft dump would produce a
      clean-looking negative result for the wrong reason.
- [ ] First random-draft training run. Nothing has trained on one yet.
- [ ] Sweep `--dg` offline. The belief reaches the network as a fixed-width sum
      of holding embeddings, and it is the one place a fixed width is a real
      approximation: a belief is a distribution over a space too large to
      enumerate, so unlike the value's dependence on a single holding this one
      cannot be made exact.

## Known gaps

- [ ] **The Warrior Priest is out of the game.** Units 18 and 54 are missing from
      `DRAFT_POOL` because the attribute triggers a *private* mid-round draw,
      which adds "which coin must I now play" to the private state. The fix is
      `Config { hand, fd, pending_coin }` — at most 5x the config count for the
      span of one coin play, still exactly enumerable — plus removing the clamp
      from the subgame's chance handling.
- [ ] **The horizon manufactures draws.** A game is cut at 256 coin plays and
      scored as a draw; War Chest has no draws. In order: report the draw rate
      beside every score instead of folding draws into 0.5; raise
      `MAX_MAIN_PLAYS` and check the cap rate rather than assuming; treat Greedy
      as a draw generator rather than a neutral yardstick (deterministic Greedy
      self-play times out 100% of the time). Scoring a timeout as -1 for both
      was rejected: it makes the game non-zero-sum, and zero-sum is load-bearing
      for CFR's convergence, the solver oracle, and `blend_outcome`.
- [ ] **The subgame is not quite zero-sum.** Its leaves are network values and
      nothing makes the net's value for player 0 at a leaf the negative of its
      value for player 1. `nash_conv` reports the residual for free. Predicting
      `v_0` and defining `v_1 = -v_0` would enforce it by construction and is
      the obvious experiment.
- [ ] Validate the mirror augmentation against the engine rather than against
      invariants. A `State::mirror()` in Rust would let the encoder itself be
      the oracle.
- [ ] Draw transitions are the largest remaining non-network cost (13%). A run of
      k draws is composed step by step over supports that grow ~5x each time;
      the multivariate hypergeometric gives the same answer directly from the
      parent support, needing a fallback for the mid-run reshuffle.
