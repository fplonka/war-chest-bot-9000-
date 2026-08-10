# v5_aliased_values

We were testing whether every public node needs a separate value vector for
both players. Values are computed for one player at a time, so the two player
layouts can reuse the same arena. At a chance node only the drawing player's
private configuration changes; the other player's value vector is exactly the
sole child's vector. The candidate aliases that unchanged vector and omits its
backpropagation copy task, mirroring the earlier reach-vector optimization.

All 17 release CUDA/library tests passed, including the learned CPU-reference
comparison, zero-network oracle, wave-composition test, and exact admission
check. A small structural test also verifies that an opponent chance node and
its child share a base while the drawing player retains distinct storage.

On the identical one-card frozen tape, interleaved 20-second runs measured
625.8 and 621.0 solves/s for the previous build, versus 641.2 and 636.4 for the
candidate. The means are 623.4 and 638.8 solves/s, a 2.47% gain. The tape's
largest job is only about 67 MiB, so this result measures reduced CFR work; it
does not yet measure whether the smaller value arena moves mature searches out
of the four-gibibyte card-exclusive class.

At this point the candidate retains the safe card-wide route introduced after
the failed long run. The next diagnostic should use the strongest trained
checkpoint and zero horizon payoff, measure late memory classes, and determine
whether value aliasing reduced the expensive route frequency before changing
the scheduler itself.
