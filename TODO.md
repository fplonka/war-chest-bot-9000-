# TODO

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
