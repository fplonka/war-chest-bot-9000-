# elo01 — the first run measured as a rating curve, and the first at 64 search iterations

**Date:** 2026-08-04 · **Result:** the agent stops improving after about
seventeen minutes · 73 ReBeL epochs in 30 minutes

## What we were trying

Two things at once, both about how we find out whether training is working.

Until this run, a run judged itself while it ran. Every twenty minutes the live
network played a few hundred games against the best network so far, and replaced
it if it won more than 55% of them. Whatever survived that ratchet was what the
run shipped. The trouble is that a few hundred games can only measure a win rate
to about three points, and two networks twenty minutes apart differ by less than
that, so most of what the ratchet was reacting to was luck. It also cost minutes
of the training budget to do it, and it produced a single number at the end that
said nothing about *when* the agent got good.

So the run now just saves the network every six minutes and judges nothing. When
it is over, every saved network plays every other one, plus the handcrafted bot
and a bot that moves at random, and all those games are fitted into one rating
per network — the same kind of rating chess uses, where a hundred points of gap
means the stronger side wins about two games in three. The output of a run is now
a curve: strength against how long it had been training.

The second thing is the search itself. At every decision the agent solves a small
version of the game by repeated self-correction, and we had been stopping that
after 16 rounds of correction because it was cheap. Sixteen is not a solved
subgame, and a solved subgame is the entire premise of the method we are using.
This run uses 64.

## What we learned

**The agent improves fast and then stops.** Ratings, with the random bot fixed at
zero:

| when | rating |
|---|---|
| 5 min (the warm-started start of ReBeL) | 356 |
| 11 min | 748 |
| 17 min | 842 |
| 23 min | 852 |
| 30 min | 852 |
| the handcrafted bot | 174 |
| random | 0 |

Six minutes of self-play is worth 392 points. The next six are worth 94. After
that, thirteen further minutes are worth ten points, which is under half of the
±22 the measurement itself is uncertain by. **The plateau is the finding**, and
nothing in the old setup could have shown it: a ratchet that only ever asks "is
this better than that" reports a promotion or two and no shape at all.

**The measurement resolves things smaller than it is asked to.** Two of the saved
networks were taken 46 seconds apart, and they came out three points apart, which
is well inside the noise and exactly what should happen. That is a real check on
the rating fit, because nothing in it knows the two are nearly the same network.

**It agrees with the old way of scoring where the two overlap.** The last network
beat the starting one 58 games to 2, and the gap in ratings between them, 496
points, is the same gap the previous run reported in its own units (0.940 as a
win rate, which is 478 points). So the new scale is not a new claim about how
strong the agent is; it is the same claim, in units that can be plotted.

**Sixty-four rounds of search cost less than feared.** Generation runs at 2.2
games a second against about 3.8 at sixteen rounds — a factor of 1.6, not the 4
the arithmetic suggests, because during self-play the search deliberately stops
at a random point before its limit, so the average solve is half as long as the
limit. What that bought is not yet known: this run cannot be compared to the
ten-minute runs before it, since almost everything else changed too. The
comparison worth making is two runs of equal length at 16 and at 64, rated on one
ladder. That is written down in `TODO.md`.

**A negative result about the games we lose to the clock.** A game is abandoned
as a draw after 256 coin plays. Against the random bot, the starting network drew
17 of 60 games that way; by seventeen minutes in it drew 3, and by the end 0 to
3. The agent is learning to actually finish games, and the invented draws are
mostly a property of weak play on the board rather than of the horizon being too
short. Against the handcrafted bot they never went away, which matches what was
already known: that bot stalls, and games against it time out by construction.

**One thing we cannot explain yet.** Over the run, decisions solved per second
fell from about 320 to about 200 while each epoch kept taking the same 22
seconds. Games were ending sooner — 137 decisions per game at the start, 92 at
the end — so the agent was winning faster, but each remaining decision was
getting more expensive. Whether that is the mix of decisions shifting toward
crowded mid-game positions, or something else, has not been measured.

## State of the project at this point

Starter matchup only (the fixed recommended draft), thirty minutes, eight cores,
search depth 2, sixty-four search iterations, five minutes of warm start. The
replay buffer holds 171,631 positions at the end — far short of the two million
it is allowed, because half an hour at this search cost does not generate that
many. Data, not memory, is still the limit.

This run is also the first with the new replay sampler: half of every training
batch is drawn from the newest fifth of the buffer and half from all of it, so a
recent position is seen six times as often as an old one. The reason is that a
target recorded long ago was written by a network that has since moved. It was
introduced in the same run as everything else, so nothing here isolates its
effect, and it has not been measured on its own.

The engine and the belief tracking are unchanged and still verified against 1,112
real games. The two known departures from the real game are unchanged too: the
Warrior Priest is still missing from the draft pool, and games are still cut off
at 256 coin plays. Both are in `TODO.md`.
