# feat01 — first run on the reworked position encoding

**Date:** 2026-08-04 · **Result:** vs Greedy 0.975, vs initial 0.953 · 115 ReBeL epochs

## What we were trying

Check that a batch of changes to how a position is described to the network did
not break anything. Two of those changes were bug fixes. The "round number"
feature divided by 40, but games routinely reach round 80 to 120, so the feature
was stuck at its maximum for most of nearly every game and told the network
almost nothing. The "stack height" feature divided by 3, but a stack can hold
five coins, so tall stacks all looked identical. Both were found by measuring
the real ranges (`engine/examples/featstats.rs`) rather than by reading the code.

We also added a description of what each unit card actually does (its tactic and
its special rules) and a marker for which piece on the board still owes a move
after a trigger. And we added, then later removed, a map of how far each square
was from the nearest piece.

## What we learned

Nothing broke. The scores land inside the normal spread for this configuration
(0.963–1.000 and 0.925–0.960 across identical earlier runs), so the encoding
change is neutral on strength at this length, which is what we wanted — a
10-minute run cannot resolve a small improvement anyway.

The wider description costs 7.6% of generation speed, measured properly by
running the old and new builds on an identical workload.

## State of the project at this point

The agent trains in about ten minutes to roughly 0.99 against the handcrafted
reference bot. Generation speed had already been optimised roughly tenfold. The
open question was what to change next before a much longer overnight run.
