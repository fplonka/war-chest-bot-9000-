# v5_reserved_whale_90s

We were testing whether keeping one of the five GPU lanes empty of ordinary
work would make it immediately available for the rare four-gibibyte searches.
Those searches still borrowed and drained one least-busy ordinary lane for
memory safety. This repeated the 90-second, strong-checkpoint, zero-payoff
trace used for the preceding routing measurements.

The reserved lane did not help overall. Pre-stop throughput was 988.9 solves/s,
down from 1,011.3 for the same configuration without lane reservation and
essentially back to the old whole-card route's 989.6. The ten guarded searches
waited an average of 1,492 ms for their two lanes and as much as 2,715 ms.
Some waits were shorter, but the helper lane could still have a long queue, and
running ordinary work on four lanes instead of five cost more than was saved.
This is a live same-seed comparison, not a deterministic zero-network tape, so
the exact percentage is directional; the result is nevertheless not promising.

All oversized searches used the two-lane guard, with no full-card routes,
exact fallbacks, or dropped solves. Sampled peak memory was 22,265 MiB on GPU 0
and 17,931 MiB on GPU 1. The project still has a safe two-lane route for these
large searches, but permanently reserving its whale lane should be reverted.
The next useful direction is to let ordinary dispatch continue while a guard
waits, rather than removing a lane from ordinary work for the entire run.
