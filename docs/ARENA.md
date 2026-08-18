# The arena

How one agent is measured against another.

## A bot is a directory

```
bots/v4-12h/
  bot           the binary, built at the revision that trained these weights
  weights.bin   the flat export train/export_weights.py writes
  bot.json      name, provenance, how it searches
```

Nothing outside the directory is ever needed again. The binary is built once
and never rebuilt, so an architecture keeps playing after the source that
produced it has been rewritten — which is the only way to answer "is the
replacement actually better than what it replaced".

`bot.json`:

```json
{"name": "v4-12h", "sha": "a661688", "mind": "rebel",
 "weights": "weights.bin", "minutes": 725,
 "search": {"depth": 2, "iters": 64, "cfr": "dcfr"},
 "note": "d2t64_long12h final"}
```

`mind` is `rebel`, `greedy`, `random` or `lbr`; one binary serves them all, so
the handcrafted reference, the floor and the exploitability probe are bots like
any other. `minutes` is how
long the checkpoint trained, and is what the run dashboard plots against.

Bots are build products, not source: `bots/` is not in git.

## The referee owns the game

The referee holds the true position and the dice. A bot holds nothing but its
own head. Between them passes only public information, plus whatever is private
to the bot being spoken to — so a bot cannot read its opponent's hand, and two
bots from unrelated revisions need agree on nothing except this protocol and
`rules_table_hash`, which the referee checks before the first move.

Two independent streams of JSON lines:

```text
referee -> {"go":    [{"id": 7, "start": {...}, "obs": [...]}],
            "watch": [{"id": 9, "obs": [{"kind": "act", "player": 0, "key": 812}]}],
            "drop":  [3]}
bot     -> {"done":  [{"id": 7, "action": 8452}]}
```

* `go` — you are to move in these games.
* `watch` — your opponent is to move in these. See below.
* `obs` — what has happened since you were last spoken to: your own draws with
  the coin, everyone else's without it, and every action reduced to its public
  observation, so a face-down play stays face down.
* `done` — these games are answered, and `action` is present for the ones the
  bot was asked to move in. A watch is acknowledged without an action, which is
  how the referee knows the bot is ready for what happens next.
* `drop` — these games are over; forget them.
* `policy` — on an ask, "send the whole strategy, not just the move". A bot
  works out a probability for every action of every hand it might hold and then
  samples one row, so the rest is already in hand. Nothing in a ladder asks for
  it; the probe and the suite do.

The two directions are not in step. The referee sends work for any game it is
not already waiting on, and the bot answers each game as that game is ready, so
neither side waits for the other's slowest game. That is where the throughput
comes from: a bot always holds a mixture of games, some having their subgame
built on its cores and some being solved on its device.

Exactly one ask per game is outstanding at a time, which is what keeps a game's
observations in order without the bot having to sequence anything. For the same
reason the referee will not tell a bot to forget a game it is still answering
about, even after that game has been decided.

A watch is a solve, so it is asked for once per position and not again until
the position moves. Without that the referee would keep handing a waiting bot
the same node while its opponent thought about it, and the bot would keep
solving it — which costs everything and buys nothing.

`PROTOCOL` is the version both sides announce, and it is checked before the
first move. Bump it whenever a message changes shape *or meaning* — a seat
receiving its own moves back was a change of meaning, and shipping it without a
bump turned a version mismatch into a confusing error deep inside a bot. Every
bot must be rebuilt when it moves; that is the price of a frozen binary.

A bot that cannot follow a game fails the whole request, and the referee
abandons the run. Half a ladder from a bot that lost the position is worse than
no ladder. For the same reason a request with a field the bot was not built to
read is refused rather than ignored, and `PROTOCOL` must be bumped — and every
bot rebuilt — whenever a message changes shape.

## Each bot keeps its own beliefs

A bot's belief over its opponent's hand only moves under some assumption about
how that opponent chooses. The assumption here is that the opponent thinks the
way this bot does: it solves the opponent's node with its own network and
filters the result on what it saw. Against a copy of itself that reproduces
self-play exactly; against anything else it is a model, and being wrong about
the opponent is part of what a ladder measures.

Nothing is shared between the two bots, so an old bot's belief code stays its
own business and the wire carries no internal state.

That model is a second solve, and `watch` is what keeps it from costing wall
clock. The model does not need the opponent's move, only their position, so the
referee asks for it while the opponent is still thinking. Both bots then work at
the same time, one card each, instead of taking turns. Without it a ladder runs
at roughly half the speed.

## Three reports, one mechanism

The ladder answers "which of these is better". Two other commands answer
questions a ladder cannot, both on the same protocol.

```bash
tools/arena.py ladder bots/greedy bots/v4-12h bots/v5-2h --games 200
tools/arena.py probe  bots/v5-2h --probe bots/lbr --games 200
tools/arena.py tablebase bots/v5-2h --suite suites/forced
```

**probe** — what knowing the opponent's actual strategy is worth. The same
probe plays the same bot over the same games twice; the second time the referee
shows it the strategy behind every move. Nothing else differs, so the change in
score is what that knowledge bought.

Be precise about what that is. A bot with no other information models its
opponent as a copy of itself — that is how it moves its belief over their hand
(see above). The probe replaces that model with the truth. So the number is the
cost of *the modelling assumption*, and it is a lower bound on exploitability
rather than a measure of it: a bot probed by a copy of itself scores exactly
zero, because its assumption was already right, and that says nothing about how
a real best response would do against it. It is most informative aimed at a bot
the probe does not resemble.

The zero is worth having as a check: if a probe built on a bot's own weights
ever showed a gain against it, something in the belief filter would be wrong.

The probe is a measuring instrument, not a player. Put it in a ladder and it
would beat everyone by cheating, and drag every other rating with it.

**tablebase** — whether a bot converts a position whose answer is proven. See
below.

## The tablebase: questions with proven answers

A ladder says which of two bots is better. It cannot say whether either is any
good, and its numbers do not survive the pool changing. The tablebase is the
other kind of measurement, and the only one here that is *true* rather than
relative.

A position qualifies when one side wins **whatever the other does** — every
line ends, and ends the same way, within a few plies. That is decided by the
rules alone. No value network appears anywhere in the proof, so the benchmark
cannot flatter the architecture that happened to build it, which is exactly
what went wrong with the first attempt (a deeper *search* shares its network's
errors, so it measures whether a bot agrees with itself given more time).

The whole game is not searchable — the public tree multiplies by about twenty
per ply and a game runs for hundreds. A forced position does not need it: only
the forced part is enumerated, which is a handful of plies. Measured over
random play, **about one late position in sixty** carries a win provable within
five plies, and **nearly all of those survive not knowing the opponent's hand**,
because a range that late is only a few hands wide.

```bash
tools/arena.py tablebase bots/v5-2h --suite suites/forced
```

A question carries the position *and both ranges*. That is the one place a
belief travels on the wire, and it is deliberate: the proof was carried out
against those ranges — every hand each side could hold given only what is
public — so the question has to be asked against them too, or it is not the
thing that was proven. In a game a bot still builds its own beliefs and nothing
is shipped; a benchmark is not a game.

The bot is seated there, plays the winning side, and makes one move. The move
is right exactly when the win is **still forced afterwards**. No opponent plays
it and nothing is sampled, so a score is a property of the bot and the position
and nothing else, and two bots years apart get the same marking. It is one
solve per question, which is why the set can be large.

Questions are collected **stratified** by how many plies the win takes and
whether any hidden information is left. Without a quota the set fills with
one-ply wins — take the marker, game over — which every bot answers and which
measure nothing.

Read the breakdown, not the average — the average hides everything interesting.

The difficulty axis is **how many of the legal moves keep the win**, which the
generator measures for every question. One is a position with a single answer.

| winning moves | 1 | 2 | 3 | 4 or more |
|---|---:|---:|---:|---:|
| greedy keeps the win | `0.21` | `0.38` | `0.29` | `0.73` |

The ply count is *not* a difficulty scale, which was a surprise:

| plies to win | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy keeps the win | `0.13` | `0.24` | `0.24` | `0.39` | `0.44` | `0.41` | `0.58` |

Deeper mates are easier, because a win eight plies out usually has several
moves that hold it while a win in two has one. Mate in one is excluded
altogether — take the marker, game over, every bot scores it.

The other axis is hidden information: greedy keeps `0.43` of the wins where the
opponent's hand is already pinned down and `0.19` where a real range remains.
That gap is the belief machinery being measured, and a benchmark that reported
one number would have buried it.

```bash
tools/arena.py generate bots/v5-2h bots/v5-2h --games 20000 --out suites/forced
tools/arena.py generate bots/random bots/random --games 20000 --out suites/forced
```

Building a set means **watching bots play** and asking the referee, at every
late position, whether the result is already decided. That the players are
bots is the point: a bot is a frozen binary behind a protocol, so an
architecture change cannot break the generator, and the source of positions is
a choice rather than a hard-coded policy. Random on both sides gives positions
from flailing; a trained pair gives positions a real game reaches. Same code.

Which to use is a real trade. Random play is far cheaper per game and yields
far more wins, so every quota fills in minutes; self-play is slower and yields
fewer, because strong players leave fewer wins lying around.

That scarcity is the argument for it. Random play mostly stumbles into
positions where half the moves win and every bot converts them; self-play
reaches the sharp ones. Measured over the same bands, positions with fewer
than one winning move in ten are `15%` of a random-play set and `36%` of a
self-play one — more than double the share of the only positions that
discriminate.

Random play is also mildly *out of distribution* for a net trained on
self-play, so a set built from it is a generalisation test as much as a
tactical one. The two measure different things; build both.

A long run publishes as it goes, so a night's work is never one crash from
nothing.

Only one question is kept per game, and the reason is not thrift. A won
position a player *converts* ends the game and yields one or two positions; one
it *misses* stays won ply after ply and yields a dozen, every one of them from
a line that player is misplaying. Keeping them all builds a set oversampled
exactly where whoever generated it is weak — an excellent adversarial probe of
that player, and a poor yardstick for anyone else. It showed up as a bot
scoring worst of all the nets on positions from its own games and mid-field on
positions from random play.

Which is the general warning: a suite is only as neutral as its source. For
comparing bots, generate from random play or from a bot that is not under
test.

The proof itself is pure rules recursion with no network in it, so it wants
cores and not a card — the GPU is for batching network evaluations, and there
are none. It also quantifies over the *whole* opponent range: a plan that only
works against the hand they happen to hold is not forced, because the winner
cannot see that hand.

### What makes it fast

The proof is the only expensive thing in the generator, and four things decide
how expensive.

It returns the *distance* to the win rather than a yes or no. One search then
answers what asking "settled in two? in three? in four?" once per depth used to
need, and a win found at some distance tightens the cap on every branch still
to be tried. The count of moves that keep the win falls out of the same root
scan, so sharpness is free rather than a second search.

Positions are proven in *bulk*. The referee records every decision node its
games pass through and the sweep drains the queue across every core at once.
Done a position at a time, as they arrived, there was never more than a handful
in hand and the cores sat idle.

The sweep runs *inside the referee*, off the interpreter lock. Driven from
Python it held the lock for the whole search and used one core of a machine
that has seventy.

The node budget is a real dial and worth understanding. It never lets an
unproven position through — it only decides which positions are cheap enough to
settle — and the handful it gives up on cost more than all the rest together.
Cutting it from 400,000 to 25,000 loses 1.5% of positions and runs 2.6 times as
fast; those it loses are spread evenly over depth, not concentrated in the deep
ones. Spend the time on more games instead.

Together these took generation from 300 random games in 65 seconds to the same
300 in 5.3 seconds, for 1.4% fewer positions. On the box, where the sweep has
seventy-two cores to spread over instead of eight, random play generates about
`107` games a second — 8,000 games, and the 2,149 positions worth keeping out
of them, in 75 seconds.

Under self-play the proof is no longer the limit at all: the cards are, which
is where the cost belongs. There the dial that matters is how many games are in
flight, because that is what fills a wave. Measured over 200 self-play games on
two cards: 48 games took `5m29`, 96 took `4m58`, 192 took `4m39`, and 384 took
`4m45` — so 192 is the knee and the default. End to end, self-play generation
went from `0.56` to `0.72` games a second.

What it does not measure: this is the tail of a game, and only positions that
happen to be forced. A bot can convert every won endgame and still play a poor
opening. It is a floor and a check, not a verdict — read it beside the ladder.

There is a boundary on which bots can be asked at all. A benchmark question
carries an encoded position, so a bot has to be able to *read* one — and the
state representation is not frozen across architectures. Between v3 and v4 the
drawn coin moved out of the continuation and into a zone of its own, so a v3
binary reads a v5-encoded position one zone short and every byte after it lands
in the wrong field. That is not a rename and no graft fixes it honestly: a
translation that guessed would produce scores, and wrong ones. Such a bot is
still fully comparable on the *ladder*, which starts from a draft and needs no
position decoding. Rate it there.

One weakness is worth naming, because it is fixable. A bot answers each
question with one *sampled* move, so a bot that finds the win 51% of the time
and one that finds it 99% of the time both look like coin flips on any single
question, and only the average over thousands separates them. The bot already
computes a probability for every move it might make, and `policy` on the ask
already returns it. Marking the probability mass that fell on winning moves,
rather than whether one sample of it did, would measure the same thing with a
fraction of the variance — the referee would have to prove every legal move at
the question rather than only the one played, which is what generation already
does to compute sharpness.

## Running one

```bash
tools/arena.py pack runs/v5_d2_125m --snapshot final --name v5-2h
tools/arena.py pack runs/v5_d2_125m                  # every snapshot
tools/arena.py ladder bots/greedy bots/v3-trav bots/v4-12h bots/v5-2h --games 200
```

Every pair plays `--games` games as colour-swapped pairs over shared drafts,
drawn from the whole unit pool. Ratings are Bradley-Terry, quoted against the
first bot on the command line — list the reference first. Elo does not carry
between separate ladders, so a comparison worth trusting is one where both bots
played in the same run.

A ladder is reproducible from its seed. Games carry their own random streams
and never interact, so it makes no difference which order the bots happen to
answer in — the same command twice gives the same result.

The pair, not the game, is the independent trial: two games over one draft with
the colours swapped remove most of the variance the draft itself contributes.
The report gives each pairing's paired record and a sign-test p-value, which is
the number that decides whether one bot is better than another. Elo is for
ordering a field, not for settling a two-way question.

Each bot gets its own card. A ladder over several bots runs the pairings in an
order that keeps one side seated where it can, because starting a bot costs a
weights load and a kernel build.

The file is rewritten after every pairing, so a long ladder can be read while
it runs. Results land in `arena/*.json` and show up in the monitor's sidebar
next to the training runs.

Concurrency past the point where the cards are busy buys nothing. Games in
flight advance in lockstep — each one waits its turn on the device — so
doubling them roughly doubles how long any single game takes to finish while
leaving the total alone. What it costs is latency and memory: a build with two
hundred games in flight produces no results at all until nearly all of them
are done, and a ladder ends with a handful of stragglers holding the devices
open. A hundred or so is enough for two cards.

## What it costs

A bot directory's binary is built for the machine that runs it. `weights.bin`
and `bot.json` are portable; the binary is not, so a bot archived on the box is
re-frozen if it has to play somewhere else.

## Reviving an old checkpoint

The bot binary has to exist at the old revision, and old revisions predate it.
Grafting it on is a one-off per architecture: check the tree out somewhere
temporary, copy in `arena.rs`, `bot.rs`, `policy.rs` and `bin/bot.rs`, fix
whatever the intervening renames broke, build, and copy the binary into the bot
directory. The graft is thrown away; only the binary is kept. Weights must be
exported by the *old* tree's `value_net.py`, since that is what defines the
layout the old binary reads.
