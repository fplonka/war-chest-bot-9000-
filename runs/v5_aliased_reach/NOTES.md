# v5_aliased_reach

We were testing whether the GPU needed a distinct reach vector at every public
node for both players. Across an edge only the acting player's private belief
changes; the other player's vector was being allocated and copied unchanged.
The candidate gives each `(node, player)` an explicit device base, aliases the
unchanged vector to its parent, and emits forward tasks only for beliefs that
actually change. The serialized format and CPU solver remain unchanged.

The first full-wave oracle exposed an important bug: strategy averaging had
been fused into the old copy task, so removing that task also skipped averages
at some decision nodes. Moving every decision-row average into one flat pass
after the reach sweep's final barrier fixed it without restoring copies. All 16
CUDA library tests then passed, including the learned full-wave, zero-network,
and wave-composition checks.

In symmetric interleaved 20-second runs on the frozen one-card tape, the
unchanged cached-readout build measured 581.0 and 582.6 solves/s, averaging
581.8/s. The aliased build measured 628.5 and 627.6 solves/s, averaging 628.1/s.
That is an 8.0% gain on identical jobs. The optimization was retained. At this
point only changed beliefs consume reach storage or forward work; snapshot
reach retains enough tail space for the cached readout mass, and admission uses
the compressed arena size.
