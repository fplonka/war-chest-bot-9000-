# traverser

One shared value network now takes the traverser as an explicit input before
the nonlinear PBS head. Old checkpoints receive a zero traverser vector when
loaded for historical matches; there is one trainable architecture and no new
format tag.

Thirty minutes, golden8 defaults, seed 1. The run completed 1,677,596 solves
and 6,710,272 optimizer rows at 1,119 balanced solves/s. The matched base was
1,132 solves/s, a 1.1% difference. The learned traverser vector ended at 0.0345
RMS and 0.1604 maximum absolute value.

Exactly 200 colour-swapped, random-draft games were played against each prior
final at depth 2 and 64 linear-CFR iterations:

| opponent | new W-L-D | score | direct Elo gap |
|---|---:|---:|---:|
| `seat.final` | 138-52-10 | 71.50% | +159.8 |
| `zsctl.final` | 112-78-10 | 58.50% | +59.6 |
| `rowfix.final` | 128-59-13 | 67.25% | +125.0 |

The aggregate is 378-189-33, or 65.75%. The combined Bradley-Terry table in
`ladder_combined.json`, anchored at `seat.final = 0`, is traverser 158.9,
zsctl 99.5, rowfix 34.6, seat 0. All Rust, Torch/Rust parity, production CUDA,
precise CUDA, and old-checkpoint publication gates passed. The adversarial Pi
review returned `ACCEPT`. Gate passed.
