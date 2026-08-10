# v5_value_alias_cardroute_aged

We were measuring which memory route the aliased-value build actually uses on
the hard game distribution. This was a three-minute generation-only stream
from the final checkpoint of `v5_cardroute_s2_20m`, with the real zero horizon
payoff, five lanes per RTX 3090, and the production depth-two, 64-iteration
search. It was not a training gate or a speed comparison; its purpose was to
split ordinary lane-local oversized searches from whole-card barriers.

Every oversized search was card-exclusive: 50 of 50 before admission stopped,
and 53 of 53 after drain. The first appeared before 30 seconds. Cumulative
throughput fell from 2,181 solves/s at 20 seconds to 1,175/s at 60 seconds and
784.9/s at 180 seconds as the stream aged. It completed 141,312 solves before
stop, with zero exact CPU fallback and zero dropped work. Peak sampled memory
was 22,265 MiB on GPU 0 and 21,931 MiB on GPU 1; mean utilization was only
49.7% and 50.3%.

This directly resolves the ambiguity in the earlier merged counter. On this
strong-policy workload, the expensive events are not mostly cheaper lane-local
whales: all of them stop and trim an entire card. Value aliasing is still a
positive 2.47% frozen-tape optimization and keeps its correctness benefit, but
it did not remove the four-gibibyte memory class. At this point the next
scheduler change should keep a hard card-memory bound while allowing unaffected
lanes to continue through these searches.
