# odd

The antisymmetric readout, gated. `v_p(c) = s_p * W + <[u, 1], g(c) - gbar_p>`,
so a seat's belief-weighted mean value is exactly `s_p * W` and the two cancel
at every PBS, for every weight setting.

Horizon 6.4% over the last quarter, `tgt_std` 0.349 — both in the band of every
run since `seat`. Its own ladder is monotone: -86 / +167 / +431 / +549 /
**+578**.

`probe_zs` reads **0.0006 for the whole run**, first epoch to last, against the
control's 0.055 flat and `zsloss`'s 0.005 bought with a penalty weight. There is
no hyperparameter here: the readout cannot express the residual.

The cost is throughput: 844 balanced solves/s against the control's 1081. A
matched nine-minute comparison put it at 8% (1007 against 1064); at thirty
minutes the gap is wider, which is worth understanding before this becomes the
default. `head_shift` is one extra kernel per head pass and one small GEMM for
the seat means.

Cross-run rating had to move to the CPU path: the pool now mixes network shapes
(tag 3 for the old readout, tag 5 for this one) and the GPU solve service is
built for a single shape. Old checkpoints keep the readout they trained under,
so the comparison is honest.
