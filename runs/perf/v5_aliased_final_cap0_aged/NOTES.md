# v5_aliased_final_cap0_aged

This run asked whether aliased reach vectors and cached readout masses improve a
real mature solve stream, not only the frozen one-card tape. It used the learned
warm-gate checkpoint, true zero payoff at the search horizon, five ordinary
lanes, one retained whale lane, and the same seed and three-minute settings as
the direct pre-alias control.

The run produced 203,776 solves in 180.00 seconds, or 1,132.1 solves/s overall.
From 120.06 seconds to the stop it produced 59,392 solves, about 991/s. It
finished 943 games and reported 1,536 searches that hit the node cap, with no
allocation failure, fallback, or dropped solve. Peak sampled memory was 18,009
MiB on GPU 0 and 22,187 MiB on GPU 1.

The direct control before aliasing produced 184,320 solves at 1,023.9/s overall
and about 838/s over its final minute, while finishing 730 games on 1,507
node-capped searches. The new build therefore gained 10.6% overall and 18% in
the mature tail, and it advanced substantially more games rather than merely
encountering fewer hard searches. This is still a generation-only diagnostic,
not the balanced training gate; the next step is to measure training with the
same OOM-safe scheduler and then attack whichever side limits the balanced
rate.
