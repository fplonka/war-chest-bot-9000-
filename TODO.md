# TODO

## Open

- [ ] The CPU solver's node cap is 200k in generation; the GPU pool sizing
      (docs/TREE.md) must budget the same tail, and the GPU build should
      receive the cap as metadata.
- [ ] **Zero-sum is a diagnostic, not a constraint.** The subgame's leaves are
      network values and nothing makes `v_1 = -v_0`. Three v4 attempts at
      enforcing it lost: `odd` showed the reparameterised readout learns fine
      (residual `0.0006`, ladder `+578`) and died on throughput (`844` against
      `1081` solves/s). The game is zero-sum and a correct architecture should
      learn that, so v5 measures the residual (`train.py::losses` reports
      `max |v_0 + v_1|`) rather than forcing it. Revisit only if the residual
      stops falling with strength.
- [ ] **The policy head, as a search change.** The v4 measurement
      (`arch_policy`, `646` against `663` and `696`) tested it as an auxiliary
      loss only, and the warm-start run that was its actual justification was
      killed before producing a number. v5 has no policy head at all. Worth
      revisiting as something the search uses, not as a loss term.
- [ ] **Direction-aware message passing** — the first trunk experiment to run
      once the box is free. The trunk sums its six neighbours with one shared
      weight, so it sees adjacency and distance but not *heading*; Crossbowman
      and Lancer need a straight line, and lines of advance matter generally.
      Per-direction taps cost `2.7x` the trunk, three axis pairs `1.67x`, and
      the axis pairs would still express a line.

## Known gaps

- [ ] **The horizon manufactures draws.** A game is cut at 256 coin plays and
      scored as a draw; War Chest has no draws. Report the draw rate beside
      every score; raise MAX_MAIN_PLAYS and check the cap rate; treat Greedy
      as a draw generator. Scoring a timeout as -1 for both was rejected:
      it makes the game non-zero-sum, and zero-sum is load-bearing.
- [x] Draw transitions: round-start runs are now one direct multivariate
      hypergeometric per source config (see rebel.rs::DrawScratch::run),
      pinned to the old composition by test.
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
