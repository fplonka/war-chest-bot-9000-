# TODO

- [ ] Gate the D5 WIP before merge: `cargo test --test rebel_solver` (new draw-endgame oracle + pre-existing micro-endgame) and `selfplay_walk`.
- [ ] Random-draft training runs (distribution extension).
- [ ] Bigger run on rented hardware (many-core CPU / GPU) once the 50x perf levers (top-k pruning, static/belief split, smaller net) land.
