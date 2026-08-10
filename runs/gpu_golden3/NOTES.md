# gpu_golden3

## What we were trying

A second thirty-minute golden run, after three further changes: jemalloc under
the builders, a pipelined lane that keeps a wave on the card while it prepares
the next one, and a deeper in-flight window (64 outstanding solves per builder
instead of 32). On the 180-second aged generation stream those three together
were worth 983.5 -> 1,119.9 solves/s, so the expectation was a golden run
somewhere near 1,150.

## What we learned

The run finished at **1,022.4 balanced solves per second** — indistinguishable
from the 1,023.5 of the run before it, and it completed almost exactly the same
number of solves (1,539,465 against 1,533,928). Zero drops, zero exact
fallbacks, 548 rows of debt, two oversized routes.

That flat result is the interesting part, and it changes how these runs have to
be read. Comparing at equal *wall time* is misleading, because a ReBeL run's
cost per solve grows as its games leave the opening: a faster build reaches the
expensive positions sooner and then spends the rest of the window there, so its
average over a fixed budget converges back towards a slower build's. At equal
*cumulative solves* — which is the same workload — this run was 27% faster at
200,000 solves and 8% faster at 400,000.

It was also 7-10% *slower* than the previous run between 600,000 and 1,000,000
solves, which is where the horizon payoff has finished annealing and the trees
are at their broadest. The likely cause is the deeper in-flight window: 64
outstanding solves per builder means twice as many trees retained on the host,
and when those trees are large that costs more than the extra pipeline depth
buys. The three-minute aged stream never runs long enough to see it. In-flight
depth should be bounded by bytes rather than by count, which is what
`docs/GPU_ARCHITECTURE.md` says and what this run is the argument for.

## State of the project at this point

Ten lanes per card, 36 builders with 128 game actors each, 64 solves in flight
per builder, jemalloc preloaded, `taskset -c 0-35`. All 17 CUDA and library
tests pass. `tools/run_rate.py` was written for this comparison and is the way
to read two of these runs against each other.
