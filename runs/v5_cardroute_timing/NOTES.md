# v5_cardroute_timing

We were measuring the exact size and wall-time cost of the whole-card routes
identified by `v5_value_alias_cardroute_aged`. This repeated the same strong
checkpoint, zero-payoff, five-lane stream for ninety seconds with a lightweight
trace emitted only for card-exclusive solves.

All 14 traced jobs needed an exactly 4,096 MiB mutable arena. Their rounded
tables ranged from 64 to 256 MiB, so total reservations were 4,160--4,352 MiB;
none approached the separate six-gibibyte threshold. The solve itself averaged
477 ms (205--1,066 ms), while waiting for every lane to drain averaged 1,574 ms
(654--2,757 ms). Across the two cards, the trace accumulated 22.0 card-seconds
of drain waiting for only 6.7 card-seconds of actual giant solves. Trimming
itself averaged under one millisecond.

The whole-card barrier is therefore substantially broader than the memory
problem requires. The next candidate should drain and trim the selected whale
lane plus one helper lane for these four-gibibyte arenas, allow the other three
lanes to keep working, and retain full-card exclusion only for reservations of
at least six gibibytes. That preserves several gibibytes of headroom on a
24-GiB card without serializing all five lanes around a roughly 4.25-GiB job.
