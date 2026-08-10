# v5_two_lane_guard_90s

We were testing the first replacement for the whole-card four-gibibyte route.
The selected whale lane and one least-busy helper lane were drained and
trimmed, while the other three lanes continued. Reservations of at least six
gibibytes still retained the full-card path. This repeated the exact
strong-checkpoint, zero-payoff, 90-second trace and seed used by
`v5_cardroute_timing`.

The routing and memory behavior were correct. All 14 oversized searches used
the new arena guard and none used the full-card route; there were zero exact
fallbacks and zero drops. Peak sampled memory was 22,265 MiB on GPU 0 and
19,595 MiB on GPU 1. Pre-stop throughput rose from 989.6 to 1,011.3 solves/s,
about 2.2%. This is a live same-seed comparison rather than the zero-network
fixed tape, so the magnitude is directional, but it is much too small to close
the performance gap.

The trace explains the small gain. Average guard drain time fell only from
1,574 to 1,359 ms. The future whale lane still carried ordinary queued work,
so the guard waited on that backlog even though the other lanes were allowed
to continue. At this point the two-lane memory guard is retained, but ordinary
work should stay off its whale lane. Four general lanes already measured within
about 3% of five on the deterministic tape; keeping the fifth lane immediately
available for rare four-gibibyte searches should trade that small opening cost
for much shorter mature stalls.
