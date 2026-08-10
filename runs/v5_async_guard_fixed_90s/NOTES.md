# v5_async_guard_fixed_90s

We were retesting the asynchronous two-lane memory guard after fixing the
queued-route deadlock found in `v5_async_guard_90s`. The selected whale lane
and one helper were blocked and trimmed for each four-gibibyte search, while
ordinary submissions continued to reach the other three lanes. A one-slot
acknowledgement buffer allowed several searches to wait on the same whale lane
without parking it ahead of an earlier release command.

The scheduler and memory behavior were correct, but the speed gain was small.
The run completed and drained normally, with 17 oversized searches, no
full-card routes, no exact fallbacks, and no dropped solves. Pre-stop throughput
was 1,023.6 solves/s, versus 1,011.3 for the synchronous two-lane guard: about
1.2% on this live same-seed comparison. Sampled peak memory was 20,121 MiB on
GPU 0 and 22,187 MiB on GPU 1.

Average guard drain time was 1,279 ms, down from 1,359 ms synchronously, and
several queued searches had effectively no additional drain wait. The searches
themselves averaged 584 ms and sometimes took 1,351 ms. Keeping three lanes fed
during a route is therefore sound but does not remove the main mature-workload
cost. At this point the asynchronous guard is retained as a modest win; the
next measurement should profile the oversized solve itself rather than adding
more routing machinery.
