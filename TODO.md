# Reassess later

**Growth stops at the round boundary.** A solve does not expand nodes past the
draw that starts the next round. The reason is cost, not soundness: the draw
re-broadens the config support, so every node beyond the boundary carries
several times the configs of one before it, and the value network is defined at
exactly that boundary state anyway. SoG itself has no depth limit; this is
DeepStack's street-boundary device. Revisit once a solve is instrumented for
frontier depth and frontier support — if past-boundary nodes turn out cheap, or
if the network prices post-draw states badly, take the stop out.

**Leaf opinions are queried once a round.** A round of `batch` regret updates
queries the network once for every leaf and reuses the per-config opinion
`v(c | beliefs)`, rescaling only by the opponent's reach mass each update
(`batch = 1` is SoG's per-iteration query). Measured inside seed noise on
every quality column at `batch = 8` — but with `snap_02`, a net that had not
learned much and may simply ignore its belief input. Re-run the `budgetq`
sweep (`ml_refresh.md` method) with a late snapshot from a real run; if the
opinion has become belief-sensitive, shrink the round rather than add a
mechanism.

## One regret rule, not two (smell)

Packed bots play with `cfr=dcfr` (+0.09 equal-weights, measured) while
training targets are generated with the simultaneous `sog` rule — two rules
in the codebase, both load-bearing. Training on dcfr measured 0.505 vs the
30-min control (neutral, possibly barely positive), so unification on dcfr
everywhere may be free. After the memory-corruption fix and the restarted
sweep settle a baseline: 30-min run trained AND played with dcfr vs the
train-sog/play-dcfr combo, 200 games. If within noise, unify on dcfr and
delete the second rule; if the split genuinely wins, document why it is
principled (paper-faithful targets, discounted play) rather than accidental.
- Re-evaluate search past the round boundary (`rounds=0` today). War Chest is more chess than poker with a small hidden state, so deeper GT-CFR trees are likely worth their cost; measure at equal compute before changing the default.
- Steady-state probe protocol: snapshots carry no replay buffer, so a resumed run trains on an empty ring for ~9 minutes and a 30-minute continuation sees ~322k fresh rows. Serialize the Buffer (arenas, offsets, counters) into every snapshot, one path, then probe changes by resuming the 125-minute reference for 30 minutes against a same-protocol control.
- Box queue: CPU-only test jobs hold the GPU flock (both GPUs idle for the length of a full `cargo test`). Give CPU jobs their own lock so they queue behind each other but not behind, or in front of, GPU jobs; only the driver sets queue priority.
