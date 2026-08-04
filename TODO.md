# TODO

- [ ] Gate the D5 WIP before merge: `cargo test --test rebel_solver` (new draw-endgame oracle + pre-existing micro-endgame) and `selfplay_walk`.
- [ ] Random-draft training runs (distribution extension).
- [ ] Bigger run on rented hardware (many-core CPU / GPU). The static/belief
      split has landed (`docs/PERF.md`); generation is ~9-10x faster end to end
      and ~26x on positions with large belief supports.
- [ ] Draw transitions are the largest remaining non-network cost (13%). A run
      of k draws is composed step by step over supports that grow ~5x each
      time; the multivariate hypergeometric gives the same answer directly from
      the parent support, needing a fallback for the mid-run reshuffle and an
      oracle against the chain.
