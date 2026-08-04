# final_check — the settings chosen for the overnight run, tried together

**Date:** 2026-08-04 · **Result:** vs Greedy 0.983, vs initial 0.932 · 95 ReBeL epochs

## What we were trying

Every change had been measured on its own. This run was to confirm the whole
combination works end to end and does not crash: the mirrored-board data trick,
sixteen search iterations instead of eight, a much larger memory of past
positions, and a shorter warm-up phase.

## What we learned

It works, and the score lands inside the normal spread for this configuration.

Worth being clear about what this run does *not* show. Sixteen search iterations
is not visibly better than eight over ten minutes, and it should not be. More
iterations cost data, which hurts a short run, and the benefit is about where
the network eventually settles over a long one. "Not worse" is the result we
were after here.

The reason for sixteen is separate and was measured directly. At eight
iterations the search's estimate of a position is slightly lopsided — a little
too high, more often than too low — and a lopsided error does not cancel out. It
feeds back into the network's own future estimates. At sixteen that lopsidedness
disappears, and past sixteen the leftover error is just random scatter, which
the network averages away by itself. So sixteen is the point where the harmful
part of the error goes away, and paying for thirty-two would buy nothing.

## State of the project at this point

Chosen settings for the long run:

    --iters 16 --cap 2000000 --warm-minutes 5 --gate-every 1200 --gate-vs both

The one thing still unverified is whether the combination keeps improving over
hours rather than minutes. The failure that would waste a whole night is the
selection gate quietly ceasing to promote new versions after the first hour.
