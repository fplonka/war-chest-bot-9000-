# cfgvalue01 — the first run after the value function was rebuilt

**Date:** 2026-08-04 · **Result:** vs Greedy 0.961, vs initial 0.940 · 75 ReBeL epochs

## What we were trying

Everything a player hides in War Chest is two piles: the coins in their hand,
and the coins they have buried face-down. Together those say exactly what is
left in their bag, which is what they will draw next round. The old evaluator
was told only the hand. So two situations where you hold the same three coins
but have buried different things — and will therefore draw completely different
coins next round — looked identical to it.

That is worse than it sounds. The search builds its plan out of the evaluator's
numbers, so when the evaluator cannot tell two situations apart, the search
hands both the same plan. Measured on the old build: 92% of such pairs got a
bit-identical set of move probabilities. The bot could not act on something it
knew. That is not an inaccuracy, it is a different game — one in which players
are forbidden to remember what they buried — and the whole self-play loop was
converging, faithfully, to that game's answer.

This run is the first on the rebuilt version, where the evaluator is asked about
one exact hidden situation at a time rather than about a hand.

## What we learned

**The rebuild does what it was supposed to.** The share of same-hand pairs that
now get *different* play went from 8% to 91%, and the typical size of that
difference went up twenty-fold. The bot uses the knowledge now.
(`cargo run --release --example cfgvalue`.)

**The score did not move, and we did not expect it to.** 0.940 against the
starting checkpoint sits inside the 0.925–0.960 that three runs of the previous
build spanned; 0.961 against the handcrafted bot is a little below its 0.99–1.00,
on 75 epochs against ~95. Two reasons that is the honest expectation rather than
a disappointment. The evaluator's own error is around 0.09, and the error the
old design was forced into by ignoring the buried coins measured 0.002–0.02 —
five to forty times smaller, invisible in ten minutes. And generating games is
about 18% slower now, because the evaluator answers per situation instead of per
hand, so a fixed ten minutes buys fewer of them.

**This run's throughput numbers are contaminated** and should not be quoted: the
machine was compiling for part of it, and generation times in the middle of the
log are inflated by roughly half. The clean figure came from `rebelbench` on an
idle machine: 12.2 games/s, against about 14 for the old build.

**One thing was much slower than expected and got fixed during the work.**
Folding the belief into the network is now a weighted sum of learned vectors
rather than a handful of counters, and written the obvious way it was 41% of all
CPU — the compiler leaves that kind of loop entirely unvectorised. Hand-written
it is 9%. Same lesson as the LayerNorm in `docs/PERF.md`, in a new place.

**A negative result worth recording:** deduplicating identical hidden situations
is not an optimisation, it is the difference between the thing working and not.
The same situation recurs at hundreds of points in one search, and at thousands
of points in one training batch. Computing its embedding once per occurrence
made the training step eight times slower; computing it once per distinct
situation put it back.

## State of the project at this point

Starter matchup only (the fixed recommended draft), ten minutes, eight cores,
search depth 2, sixteen search iterations. The engine and the belief tracking
are unchanged and still verified against 1,112 real games.

Two things about this agent are still not the real game, and both are now
written down in `TODO.md` rather than assumed to be fine. The Warrior Priest is
missing from the draft pool — two of nineteen units — for the same kind of
reason the hand-keyed evaluator existed: its ability makes a player draw a coin
mid-round in secret, which adds a third thing to the hidden state, and the unit
was deleted instead of the hidden state widened. And the game is cut off after
256 coin plays with an invented consolation score, which is faded to zero part
way through training and is zero for all evaluation, but is not gone.

The one place where an exact answer is genuinely unavailable: the *belief* — the
spread of possibilities you think your opponent might be in — reaches the
network as a fixed-size summary. Unlike a single hidden situation, a probability
distribution over all of them cannot be written exactly at fixed size, because
there are about 145,000 of them. That is ordinary approximation, it does not
change which game is being solved, and its cost has not been measured yet.
