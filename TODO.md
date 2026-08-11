- [ ] experimetn with turbo vs not
- [ ] **The subgame is not quite zero-sum.** Its leaves are network values;
      predicting v_0 and defining v_1 = -v_0 would enforce it by construction.
- [ ] Validate the mirror augmentation against the engine rather than against
      invariants (a State::mirror() in Rust would make the encoder the oracle).
