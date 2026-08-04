# feat02 — encoding without the distance map, and the dump we analysed

**Date:** 2026-08-04 · **Result:** vs Greedy 0.990, vs initial 0.945 · 106 ReBeL epochs

## What we were trying

Two things. First, confirm that removing the distance-to-nearest-piece map cost
nothing. That feature was a summary we had computed ourselves rather than a
plain fact about the position, and it happened to be the same quantity the
handcrafted reference bot uses — so it was quietly teaching the network that
bot's opinion. We would rather the network work things out from the raw
position.

Second, produce a saved copy of the training data for offline analysis.

## What we learned

Removing the distance map cost nothing: 0.990 and 0.945 are as good as with it.

The saved data from this run is what settled the main question of the session.
Held-out error would not drop below about 0.09 no matter what we did, and we
tested three explanations for that. It was not a bug — positions that are
byte-identical produce byte-identical answers, so nothing is inconsistent. It
was not that the answers go stale as the network changes during a run; that
effect is real but far too small. And it was not that the network was too small;
five different network shapes spanning a 2.6x range in cost all performed within
4% of each other.

The actual answer was that the network was **short of data**, not of size. It
was scoring far better on positions it had already seen than on new ones. More
data helps and shows no sign of levelling off.

That led directly to two changes: presenting every position a second time by
rotating the board 180 degrees and swapping the two players, and making the
memory of past positions much larger.

## State of the project at this point

Everything above was measured offline on this saved data, which is far more
reliable than comparing training runs — a ten-minute run's score wanders by
about ±0.05 on its own, which is larger than most of the effects we care about.
