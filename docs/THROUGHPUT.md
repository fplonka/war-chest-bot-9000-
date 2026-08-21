# 150 solves a second: what has to be true

The target is **≥150 balanced solves/s** at `nodes=8192, expand=8, iters=64`
over a real 30-minute `train.py` run on the two-3090 box, with Student of Games
implemented faithfully. The algorithm is in. What is left is throughput.

## Where the rate stands

| when | measured on | solves/s |
|---|---|---:|
| run `gpucfr2` | real 30-min train.py | 9.8 |
| run `cohorts10` | real 30-min train.py | **21.8** |
| after the sweep and trunk work | `farmprobe`, 366 s mean | 28.0 |
| after power-of-two arenas | `farmprobe`, 240 s mean | **32.5** |

**Take the run, not the probe, and never take one probe window.** A solve's
cost varies twenty-six fold with how far into a game its root sits, so a short
window is a sample of the mix rather than a rate. Over one six-minute probe the
sixty-second windows read 44.8, 5.3, 40.1, 27.3, 23.1, 26.8 -- an eight-fold
swing around a mean of 28.0, while rounds a second stayed between 39 and 56.
Only the long mean means anything. What is stable window to window is *rows a
second*, which held near 180,000 throughout; rank two builds on the kernel
table and on rows, and confirm on a real run.

A cohort is **not** marching in step, which was the obvious suspicion: a solve
is a fixed forty-two rounds and every round advances every solve by one, so
threads that start together could stay together. `tools/convoy.py` reads the
per-launch grid of a sweep out of an nsys capture and buckets it by time; over
25 s the backward sweep swings 1.4x with no periodicity and the trunk 2.5x. The
swing in completions is the cost mix, not a convoy.

## The arithmetic

```
rate = threads / (rounds_per_solve × round_time)
```

A solver thread runs one solve and parks at a barrier once per round; a round
runs when every live thread has parked. Measured: `72 / (64 × 114 ms) = 9.9`,
against a real run's 9.8. All three factors have to move.

## Two ceilings, and both are close

**Arithmetic.** A solve is ~200-310 GFLOP, of which the trunk is roughly half.
At 150 solves/s that is **30-47 TFLOP/s**. Two 3090s give 71 TFLOP/s of FP32 —
so the target is 42-66% of peak on GEMMs shaped `[cells, 192] × [192, 96]`,
which no library reaches. In FP16 with FP32 accumulate the peak is 142 and the
target is 21-33%, which is ordinary. **The tensor cores are not an
optimisation; they are a precondition.** The pre-SoG implementation used FP16
head GEMMs by default and the rewrite dropped them.

**Bandwidth, and this is the tighter one.** A trunk residual block is written as
eight kernels — two norms, a neighbour mix, two GEMMs, a pool, a group bias, an
accumulate — and every one streams the whole `[37 hexes, 96 channels]` board
through global memory. That is ~12 passes over 14 KB per block, ~1.4 MB per
trunk row over eight blocks. At 150 solves/s and ~5,700 leaf rows a solve that
is **~1.2 TB/s against the 1.9 TB/s the two cards have**.

A whole board is 37 × 96 × 4 = **14.2 KB**. It fits in shared memory, and the
weights of all eight blocks come to 1.5 MB, which fits in L2. A trunk that
loads a board once, keeps it resident and returns it once moves ~24x less
memory than one that does not.

## What the profile says now

nsys, 25 s, both cards, ten cohorts, after everything below:

```
GPU span        25.0 s
kernel busy     16.6 s   66% of span
idle gaps        6.8 s   27%, p50/p90/p99 2.8 / 131 / 2,649 us
launches       384,205
```

| share | kernel | launches | avg us | avg grid |
|---:|---|---:|---:|---:|
| 23.9% | `k_trunk` | 2,313 | 3,557 | 1,342 |
| 17.1% | `k_readout` | 10,823 | 542 | 21,726 |
| 14.7% | `k_backprop_sweep` | 69,872 | 72 | 820 |
| 11.0% | `ampere_sgemm_128x64_nn` | 49,385 | 77 | — |
| 9.9% | `k_belief_pool` | 10,825 | 315 | 21,725 |
| 7.4% | `k_reach_sweep` | 89,468 | 29 | 902 |
| 5.2% | `k_norm` | 43,295 | 41 | 2,716 |
| 3.6% | `k_expand` | 5,916 | 206 | 12.6 |

Each of these is a different distance from its own roof, and knowing which is
what decides the work:

* `k_readout` moves ~22 config rows of `D` floats a query and reaches
  **645 GB/s, two thirds of a card's bandwidth**. It is at its roof. The only
  thing that helps is fewer bytes, and half precision is barred -- see below.
* `k_belief_pool` does the same shape at 285 GB/s, so it has room: it is the
  `cidx` then `g` double indirection, not the bytes.
* `k_trunk` is arithmetic and now at 2.65 us a row.
* The two sweeps together are 22% of device time over **159,000 launches**, and
  move about 333 MB a solve -- **forty-four times off the bandwidth roof**. They
  are latency, and the shape of the fix is fewer dependent loads a cell.
* `k_expand` runs **twelve blocks** for 206 us. Its trajectories are sequential
  by definition, so it is one warp a solve chasing pointers down the tree.

Against the earlier capture the trunk has gone from 73.2% of device time at
66 ms a launch to 28.7% at 5.2 ms -- **12.6x**, from putting its accumulators
back in registers. It is now at 4.2 TFLOP/s, **12% of a card's FP32 peak**,
against an issue-bound ceiling around 16 and an L2-bound one around 22. So
there is still 3-4x in that kernel before precision is the question, and it is
latency, not arithmetic: two blocks an SM is fifty per cent occupancy for an
inner loop of three global weight loads and four shared reads per twelve fused
multiplies.

The other shape worth naming: `k_reach_sweep` and `k_backprop_sweep` are 24% of
device time across **259,000 launches** averaging 23 and 80 microseconds. That
is the per-level sweep, launched once a level a traverser a iteration a cohort.

## Where a round's time actually goes

nsys says GPU kernel busy is **7.7%** of the span. The `prof` build says the
solver threads are parked ~100% of their time. Per card, one driver thread runs
the whole of `Card::round`.

The cause is not the marshalling alone. **A copy from ordinary pageable host
memory is not asynchronous**: the driver stages it through a pinned buffer of
its own, blocking the calling thread and draining the stream. A round issued
about ninety of those — three in the trunk, two per trunk call for the belief
index, three in the config encoder, four in the scatter, five per batch layout —
so a round with one explicit synchronise had ninety implicit ones, and the cards
stood idle through all of them.

## Order of work

1. **Per-solve CFR factors.** *Done.* `Card::iterate` read the decay factors
   from the first call of its shard and applied them to every solve in it, but
   the factors are a function of that solve's own iterate count. Every solve but
   one was running with another solve's discount weights, in every batched run
   to date. `cuda_parity` missed it because it ran one solve at a time;
   `a_solve_does_not_depend_on_the_round_it_rides_in` covers it now, and shows
   targets agreeing to 6e-6 across batch compositions.
2. **Page-locked staging.** *Done.* Every upload a round makes goes through a
   kept page-locked buffer, and every write to a solve's arrays — the belief
   index included — travels in the round's one scatter. There is no pageable
   upload path left. GPU utilisation went from 7.7% kernel-busy to ~70%.
3. **Marshalling off the driver thread.** *Done.* A solve now hands over its
   writes already concatenated and already in words: `Contract::write_into`
   builds them on the solver thread, and the driver only says where each run
   lands. `the_runs_a_solve_sends_rebuild_its_contract` pins that the delta a
   solve sends reconstructs its contract exactly, without needing a GPU. The
   tail a growth appends is sent once and written to both `cur` and `prior`,
   which used to be two copies of the same numbers.
4. **One iteration a round.** *Done.* Riding a solve's whole tail in one call
   saved most of its barriers, but a round holds thirty-odd solves and at most
   one is in its tail, so the extra iterations ran over one solve's leaves
   instead of thirty-six. The barrier is the cheaper of the two once a round's
   marshalling is off the driver.
5. **A fused trunk.** *Done, and the largest single find.* The board stays in
   shared memory for all eight residual blocks. Fusing it was not enough on its
   own: nsys put `k_trunk` at 73% of device time running at **3% of the card's
   FP32 peak**, because its accumulators were in local memory rather than
   registers. Two causes, both invisible in the source. A loop whose trip count
   the compiler cannot see forces a local array to memory, so `TRUNK_SPAN` and
   `TRUNK_MAXH` are compile-time and the odd hex is masked rather than skipped.
   And nvcc caps registers to fit more blocks per SM and spills the rest, so
   the kernel carries `__launch_bounds__(384, 1)` -- shared memory already
   limits it to one or two blocks an SM, so there was nothing to trade.
6. **FP16 GEMMs.** The arithmetic ceiling above.
7. **Overlap.** *Done.* Two cohorts of solves, each with its own gate, driver
   thread and lane of the card -- a lane being a second copy of a card's
   working state on a stream of its own. The driver is busy ~90% of a round and
   only a third of that is waiting for the card, so the cohorts fill each
   other's gaps: 15.8 to 27.3 solves/s, and the farm's `device` share reads
   over 100% because two drivers are inside the backend at once.
8. **Solves in flight.** *Done.* Two things bounded it. The leaf pass's
   intermediates are 5,640 bytes a leaf row and were sized by the whole round,
   which is a gigabyte a lane; they run a tile of leaves at a time now, at
   ninety megabytes. And `avg` was a second copy of `sum`, written once at the
   end of a solve and read only after -- so it is the same array.

**Cohorts are the lever, but there is an optimum.** All at 288 solves in
flight:

| cohorts x threads | solves/s |
|---|---:|
| 8 x 36 | **55.7** |
| 12 x 24 | 53.0 |
| 6 x 48 | 32.0 |
| 24 x 12 | 35.1 |
| 16 x 18 | 25.7 |

More cohorts of fewer solves beats fewer of more, up to a point: a cohort's
round pays its barrier once, so a shorter round means a cheaper barrier and
finer-grained overlap. Past thirty-odd solves a round the batch stops being
worth an accelerator's attention and the driver threads start outnumbering
anything the cores can serve. Thirty-six is the shape to hold. With the shape fixed, the count peaks at ten:
eight gives 50.5, ten 60.3, twelve 53.6.

At ten cohorts a driver's round divides as **41% `download` and 59% host**.
`download` is the blocking `memcpy_dtov` that ends a round, so it is the card;
the rest is work on the one thread a lane has:

| share of a round | stage | what it is |
|---:|---|---|
| 41% | `download` | waiting for the card |
| 28% | `trunk` | of which 18 points is staging `xpub` |
| 12% | `hand-back` | splitting replies back to solves |
| 11% | `tree` | placing a round's writes |
| 7% | `configs` | the config encoder |

The card is *not* the largest single item. Fifty-nine per cent of a round is
one host thread, and a solve waits through forty-two of them.

Note what this does **not** mean. Host work a round is not fixed: it is
proportional to the growth the round describes, which is why `grow_every` below
bought nothing. What has to fall is host work **a solve**.

## The ceiling nobody was watching: device memory

At ten cohorts of thirty-six both cards read **24,027 MiB of 24,576**. Solves
in flight is what the rate is linear in, and it is already at the wall -- which
is also the best explanation for why the rate is erratic rather than merely
noisy. Once a stream-ordered pool cannot satisfy a request from what it holds,
it synchronises and reclaims, and a full-device synchronise inside a round is
exactly the shape of the 2.6 ms gaps at the ninety-ninth percentile.

`leaf_breakdown` reports `held`, the live bytes every solve arena holds. It is
a level, not a rate: `Arr::fit` adds what it takes and `Arr::reset` gives it
back.

`Arr::reset` looks like the culprit and is not. A slot that kept its high-water
mark would need the worst case times the number of slots, and a solve's cost
varies twenty-six fold, so freeing between solves is right. What is suspect is
the size a solve asks for. `grow_to` returned `2 * want`, and `want` is a cell
or node count, so the sizes are arbitrary: one solve gives back 260,002 floats
and the next asks for 259,884. Stream-ordered free does not return pages to the
driver -- it returns them to a pool, where a block can only serve a request it
is large enough for -- so the pool holds both, and keeps holding, until it has
the sum of every size any slot ever wanted. Rounding to a power of two costs
the same average slack and makes every block interchangeable.

**The class has to be coarse, and that is not obvious.** A power of two leaves
up to half an array as slack, and the census says that is real: the fattest
solve holds 179 MB with nine of its arrays at exactly 2^21 floats. Eight
classes to an octave bounds the slack at an eighth -- and ran both cards out of
memory at twelve cohorts, where powers of two had held. A retained pool does
not care how much slack a block carries; it cares whether some other slot can
use it, and eight times as many classes is eight times fewer blocks that fit.
**Reuse beats slack.** The test to keep is not that the slack is small but that
an octave holds exactly one class.

**Measured.** Peak memory went from 24,027 MiB pinned at the ceiling to a peak
of 17,969 settling near 16,000, and the rate from a 28.0 mean to **32.5**.
`held` now reads 17,212 MB against the same 16-18 GB nvidia-smi sees, so what
the cards hold is live arenas and not pool residue: **47.8 MB a solve**, where
it was above seventy. The windows also stopped swinging -- 30.0 and 35.3, where
the run before read 44.8, 5.3, 40.1, 27.3, 23.1, 26.8. That is the signature of
an allocator that was reclaiming inside a round.

Two traps this exposed, both worth remembering. A probe killed by dropping its
ssh leaves the python holding the cards, and the next run dies with
`CUDA_ERROR_OUT_OF_MEMORY` at startup while nvidia-smi still reads 24 GB --
launch box work under `setsid nohup` and kill it by name. And a memory reading
taken during that teardown is not a measurement of anything.

## What the cards wait for

`tools/gaps.py` attributes every idle gap on a card to whatever kernel ran
before it. Over a 25 s capture, **22.7 s of idle across the two cards**:

| share | after | gaps | each |
|---:|---|---:|---:|
| 28.1% | `k_expand` | 1,394 | **4,571 us** |
| 16.1% | `k_backprop_sweep` | 17,140 | 213 us |
| 13.0% | `k_bag` | 1,291 | 2,292 us |
| 7.9% | `k_finish` | 3,501 | 515 us |

`k_expand` is the last kernel of a round and `k_bag` the last of the config
encoder, so those two gaps *are* the host turnaround, measured from the card:
**41% of all idle is the cards waiting for the host between phases.** The
`k_backprop_sweep` entry is different in kind -- seventeen thousand gaps of
213 us *inside* the CFR loop, between level sweeps, which is launch and
dependency latency with nothing else queued to fill it.

This is worth more than any one kernel. The cards are busy about 55% of the
time; the whole kernel table is competing for the other half.

It also reopens `grow_every`. The measurement below found it worth nothing,
and the reasoning was that halving the wakes doubles the device work a round
carries. That holds only when the device is saturated, and it is not: fewer
host turnarounds is fewer 4.6 ms holes.

## What a solver thread does between rounds

`warchest.awake()` times each thread from coming back through the gate to
submitting again -- its share of the host turnaround. Measured:

```
solver threads awake: 16.33 ms mean over 15,835 spans, longest 173.2 ms
```

**The mean is the finding.** Sixteen milliseconds a thread a round is real
work -- growing the tree, rebuilding the contract delta, building the writes --
not a scheduling artifact, and not one straggler holding thirty-five others.
The tail matters too: a round waits for the slowest of thirty-six draws from a
distribution whose maximum is 173 ms, and the expected maximum of thirty-six
such draws is comfortably over a hundred, which is the wait a round shows.

So there are two separable attacks, and the first is the one with headroom:

* **Make the mean smaller.** Sixteen milliseconds against a round of 364 is
  where the host cost lives. `describe` and `t-marshal` are the named pieces.
* **Stop waiting for the tail.** `round_before` already lets a thread miss a
  round and join the next, and `PATIENCE_MAX` is 50 ms, so in principle the
  tail is already cut off. That it does not appear to be is worth chasing:
  a round is 364 ms of which about 135 is `eval`, and 50 ms of patience does
  not account for the rest.

## Where the host time goes

A `prof` build attributes it. Excluding `WAIT`, which is a thread parked at
the gate:

| stage | cpu-ms | share |
|---|---:|---:|
| **PRIOR** | 162,870 | **65%** |
| PUBFEAT | 23,369 | 9% |
| CONTRACT | 19,489 | 8% |
| BSUP | 10,570 | 4% |
| BCELLS | 7,061 | 3% |
| BDRAW | 6,959 | 3% |

Two thirds of it is `refresh_priors`, not tree growth -- which is what the
shape of the round had suggested twice, wrongly.

Half of that had a structural cause. `refresh_priors` opened with a scan of
every node, every iteration, looking for the handful just grown: eight
thousand nodes by sixty-four iterations is half a million filter tests over
four scattered arrays, and it grew with the tree. Growth knows which nodes it
made, so it queues them and the refresh drains the queue.

What that bought, on eight threads and one card:

| | before | after |
|---|---:|---:|
| solves/s | 10.7 | **13.3** |
| a thread awake, mean | 11.34 ms | 9.56 ms |
| a thread awake, **longest** | 175.3 ms | **58.4 ms** |

The maximum is the number that matters: a round waits for the slowest of
thirty-six threads, so a tail cut by two thirds is worth more than the mean
being cut by a sixth.

**`PRIOR` is still the largest item afterwards, and now it is all network.**
What is left inside it is `Net::actions` and `Net::policy` -- the action head
and a dot product per legal cell, run *on the host*, for every grown node,
every iteration. The device already computes the config half and ships `f_p`
back for this. Moving the whole prior onto the card is the next large piece of
host work to remove, and unlike `grow_every` it changes no answers.

## What the memory actually is

`tools/farmprobe.py` prints the fattest solve a card held, array by array, off
`warchest.solve_census()`. Two captures at ten and twelve cohorts:

```
179.2 MB: vals=16.8  p=8.4  f=8.4  cidx=8.4  reach=8.4  cur=8.4  regret=8.4 ...
354.8 MB: reach=33.6 vals=33.6 f=16.8 cidx=16.8 cur=16.8 regret=16.8 ...
```

Against a mean of 65 MB. **The ceiling is set by the tail, and the tail is
five-fold.** It is not waste: `reach` and `vals` are per (node, player, config)
and per (node, config, traverser), so what makes a solve fat is the *width of
its belief*, which `config_cap` bounds at 256 a player a node. The comment in
the gate already said it -- "a round-start position with a broad belief is
worth many ordinary ones" -- and this is the same fact costing memory rather
than time.

So there are two ways to raise solves in flight, and only the second is free of
consequences for what gets searched:

* Narrower arrays. The eight float arenas are 168 MB of that 355. Half of them
  accumulate across a solve's sixty-four iterations -- `sum`, `regret` -- so
  half precision is a question about CFR convergence, not just about bytes.
* Admission control. Every slot is equal today, so one slot drawing a broad
  root costs eight ordinary ones, and the card has to be sized for all of them
  drawing one at once. A farm that knew a solve's belief width before admitting
  it could hold the *mean* rather than the maximum. It also decides which
  solves run when, so it needs care not to bias what the trainer sees.

## What is left, in order

There is no single item worth more than about 1.2x left. The profile is flat on
both sides, so 28 to 150 is the product of several, and the two halves have to
come down together: at 150 solves/s the card must do a solve's device work in
13 ms where it now takes 55, and a driver's round must stop being 59% host.

1. **One row a leaf.** *Done, pending measurement.* `encode` wrote the mirrored
   seat view of every leaf. Nothing read it: `Net::board` gathered the physical
   rows and threw the rest away, and the device driver gathered them one leaf at
   a time on its single thread. It is gone -- the card table still wants a
   mirror, so the first row's is kept and nothing else is. That halves the
   public-feature work on the solver threads, halves the call, and turns the
   driver's largest non-device stage into one copy a call.
2. **Hand-back onto the solver threads.** 12% of a round, and the same argument
   that moved the marshalling: the solvers are parked while the driver does it.
3. **Fuse the join chain.** `ampere_sgemm_128x64_nn` plus `k_norm` is 16% of
   device time in ten launches a tile, at 9.3 TFLOP/s, and every one streams a
   row of 128 floats back through memory. The trunk's own fusion is the
   template. It would also make the join batch-independent by construction,
   which cuBLAS is not -- and that is the thing that blocks half precision
   everywhere else.
4. **A deterministic config encoder**, which is what half precision for `f` and
   `g` actually needs. See the abandoned note below: the objection is not the
   precision, it is that cuBLAS sums in an order that depends on the round's
   shape. Fix that and 27% of device time can halve.
5. **The sweeps.** 22% of device time, forty-four times off their bandwidth
   roof, over 159,000 launches. Latency a cell, not bytes.
6. **The trunk's arithmetic.** Now 2.65 us a row after the lanewise weights.
   Packed half in its inner loops is next; its matrix shape is fixed per leaf so
   it carries no batch-dependence risk, but half accumulation over ninety-six
   terms will not hold `the_network_agrees` at 1e-3, so that test needs an exact
   path for the oracle and a separate bounded fast-versus-exact check.
7. **Solves in flight without threads.** Ten cohorts of thirty-six is 360
   solver threads on 72 hardware threads, and they all wake at once when a
   round ends. The pre-SoG design ran 36 OS threads each multiplexing 32
   lightweight solves, which is how it held 1,152 in flight. Device memory
   allows roughly double what is in flight now -- 144 solves held 10.9 GB, so
   about 76 MB each -- and no more.

## `grow_every`, which now buys a great deal

`rounds_per_solve` is about forty-two, and a round exists for one reason: the
host has to turn the leaves an expansion phase sampled into decision nodes,
because that is the game's rules. `Cfg::grow_every` lets a solve run several
iterations, and several expansion phases, before the host is woken.

Measured at ten cohorts of thirty-six, on cards verified free between points:

| `grow_every` | solves/s | rounds/s | rows/round | device |
|---:|---:|---:|---:|---:|
| 1 | 28.6 | 55.2 | 3,568 | 723% |
| 2 | 33.3 | 37.1 | 5,313 | 792% |
| 4 | **47.4** | 26.4 | 9,324 | 849% |

**This was measured as worth nothing before, and the earlier reasoning was
sound at the time:** halving the wakes doubles the device work a round
carries, so `rate = in_flight / (rounds x round_time)` is unchanged. That holds
when the device is saturated. It is not -- it is busy about 55%, and
`tools/gaps.py` says 41% of the idle is the host turnaround after `k_expand`
and `k_bag`. Fewer turnarounds is fewer 4.6 ms holes, and the secondary
numbers say exactly that: rounds a second fall, rows a round rise, device
occupancy climbs.

**It is not free, and it must not be adopted on the rate alone.** A round of
`k` iterations runs `k` expansion phases against a tree the host has not grown
in between, so the second and later phases select from a stale tree and can
sample the same leaf twice. The tree GT-CFR builds is a different tree. The
earlier note also recorded `L/var` rising from 0.2 to 5.5 at two, which is a
hint that the targets get noisier.

The gate this needs is search quality, not Elo: `Solver::nash_conv` already
measures exploitability of the finite search game, and `tests/rebel_solver.rs`
already holds solves to it. What is missing is that the host solve loop
ignores `grow_every` -- it grows every iteration -- so the host cannot yet
reproduce the device's semantics to be measured against them. That is the next
piece of work before the default moves.

## What was tried and abandoned

**Everything that raises memory pressure costs rate, and nothing else about
the allocator matters.** Three measurements say the same thing:

| change | effect on memory | solves/s |
|---|---|---:|
| arbitrary sizes (`2 * want`) | pool holds every size ever asked | 28.0 |
| a power-of-two class | pool reuses; peak 24,027 -> 17,969 MiB | **32.5** |
| eight classes an octave | eight times fewer blocks fit a request | out of memory |
| `RELEASE_THRESHOLD` at max | pool never gives pages back | 24.8 |

Holding the pool was meant to stop the driver re-mapping pages, which is
device-wide and would stall every lane. It made things worse, and `held` went
from 17.2 GB to 25.9. So the allocator is not a hidden serialiser; it is a
*memory* effect throughout, and bytes a solve is the lever.

Worth correcting an error that motivated that experiment: the probe's `device`
percentage is wall time inside `eval`, which includes waiting on the card, and
is **not** kernel time. Reading it as kernel time suggests the cards should be
97% busy against nsys's 66% and invents a serialiser to explain the gap. There
is no gap. The cards are 66% busy because that is the work 360 solves make.

**Parallelising the trunk's single-warp sections.** The pooled reduction and
the group-bias projection ran on one warp of twelve, the latter six hundred
multiplies and six hundred loads a lane. Spreading them over the block and
reducing in shared measured 2.68 us a row against 2.65 -- neutral -- so it is
not worth four barriers and 4.6 KB of shared.


**Half precision for `f` and `g`.** The config readout is gathered a row at a
time, once per config per leaf per iteration -- about 43 GB a solve out of L2,
the largest byte flow in the design -- so storing it as half looked obvious.
It fails `a_solve_does_not_depend_on_the_round_it_rides_in`: the root policy
moved **1.4e-1** between batch compositions against a 5e-2 bound. The mechanism
is worth remembering. The config encoder's matrix multiply has a shape that
depends on how many configs the round happens to carry, so cuBLAS sums in a
different order and `f` differs in its last bits; half storage rounds those two
values to *different* halves, a 1e-3 step rather than a 1e-7 one, and regret
matching turns that into a visible difference in the strategy. Targets survived
it, the policy did not. After the trunk was fixed the readout and the pooling
are about a tenth of device time, and a tenth is not worth a training target.

**Nsight Compute does not run on this box.** `ERR_NVGPUCTRPERM`: performance
counters are gated by a kernel-module parameter on the host, which a container
cannot set. nsys plus arithmetic is the instrument. Two traps in nsys itself:
`--duration` kills the application when it elapses, so a profiled probe never
reaches its own report; and `nsys export` will not write its sqlite into a
directory it cannot reach, silently, under `set -e`.

## How it is measured

`--cohorts` sets how many independent cohorts of solves run at once, one lane
of each card apiece. It is bounded by device memory: four cohorts of
thirty-six solves fill 15 GB of a 24 GB card, and six do not fit. Solves in
flight is `cohorts x threads`.

`tools/farmbench.py` runs the farm over a fixed corpus of roots sampled from
real play, cycled on interleaved strides, so the mix of solve costs in flight is
stationary. `tools/farmprobe.py` plays games forward instead, and a solve's cost
varies twenty-six fold with how far into a game its root sits — consecutive
probes of one build measured 16.6, 16.2, 12.1 and 8.0 solves/s. Use the bench to
rank builds; use `train.py` to claim a rate, because only a run has the
trainer's card contention, mid-run weight publishes and shifting position
distribution.

`cargo test --release --features gpu --test cuda_parity` is four tests and about
three seconds. Run it after every change and before any measurement.
