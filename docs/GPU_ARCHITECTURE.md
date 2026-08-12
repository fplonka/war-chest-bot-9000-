# Architecture for 1,200--2,000 ReBeL solves/s

Status: implementation in progress. The five-minute gate is met --
`runs/v5_c_gate_l8` reached 1,404.9 balanced solves/s -- and the first complete
thirty-minute golden run, `runs/gpu_golden`, reached 1,023.5. See
`docs/GPU_PERF_GOAL.md` for the current numbers and the aged-stream ladder that
produced them. What follows is the design; the section below records where its
diagnosis turned out to be wrong.

This is a replacement design, not an incremental plan for the resident CUDA
service. Keep the verified rules, tree semantics, compact training-row format,
network split, and CPU solver as the oracle. Replace the way work is scheduled,
represented on the GPU, retained between phases, and delivered to the trainer.

There is exactly one CUDA architecture in the tree. The v5 implementation
deletes the v4 resident service, live-set layout, tick kernels, job contract,
and their plumbing; it does not add a `gpu_v5` mode beside them, retain a v4
fallback, or build an adapter between the two. Git history and frozen CPU
oracle tapes preserve the old semantics more cleanly than dead production
code would.

The proposed system has four defining properties:

1. CPU workers build verified public trees into a sparse, immutable contract.
   They never block as game-sized threads waiting for CUDA.
2. Each GPU executes cost-bucketed, contiguous waves on reusable CUDA streams.
   Homogeneous frozen tapes can reuse CUDA Graphs; heterogeneous live waves use
   direct launches. A wave has one weight version and one fixed 64-iteration
   schedule; there is no fragmented per-solve slab or host-driven live-set tick.
3. Mutable CFR state exists only for legal action cells. Instantaneous regret
   and a persistent normalised-average arena do not exist.
4. Solves, replay insertion, optimizer steps, and weight publication are one
   continuous pipeline. There is no game-batch or epoch-wide drain.

That is the shortest design I think can reach 1,200 on the existing Vast.ai
box and still have a credible path to 2,000.

## What the measurements actually said

Four beliefs in this document did not survive contact with the machine. They are
kept here because each one cost days.

**"The device is the ceiling."** It is not, and neither is the host. Every
configuration of lane count, wave size, cost class, fill window, builder thread
count, in-flight depth, allocator and even CFR iteration count landed the live
stream within a few percent of the same number, while the cards were about half
idle and the builders used under a third of the box's threads. Two independent
processes, one per card, produced exactly the same total as one process using
both. The system was latency-coupled: builders waited on cards, cards waited on
builders, and no single resource was saturated. Nothing that adds capacity to
one side fixes that; only shortening the loop does.

**The frozen tape is not a decision metric for live throughput.** Merging the
cost classes and shortening the wave fill window are worth about +50% on
`wave_tape` and nothing at all live -- and merging classes is a 5% *regression*
on the aged stream. The tape has no game loop, so it measures the executor in
isolation and rewards changes that the live system cannot use. Use it to
attribute executor cost, not to choose.

**Memory was the hidden constraint, and not the way this document assumed.** The
arena was not the problem; *retaining* it was. A lane grew its buffers to the
largest wave it had ever served and never gave them back, so a single
gibibyte-sized search per lane filled a 24 GiB card. That is what killed the
long run at fifteen minutes, and it is why every attempt at more lanes or more
in-flight solves ended in `CUDA_ERROR_OUT_OF_MEMORY` rather than a measurement.
Returning oversized buffers took peak memory from 19.7 to 7.9 GiB at no cost in
throughput, and only then could the pipeline be widened at all.

**The idle time was one gap per convoy cycle, not launch overhead.** An Nsight
trace attributed 94% of both cards' idle to a few hundred gaps a second apart,
each around 30 ms, every one of them ending at `pack_cards` -- the first kernel
of a wave. Lanes went quiet between waves, and since they all wait on the same
card they went quiet together. Splitting `solve` into `launch` and `collect`, so
a wave stays on the card while the lane assembles the next one and unpacks the
last, is what made a deeper in-flight window pay for the first time: in-flight
80 was worth +9% after that change and nothing before it.

**The host tree builder was not short of algorithm, it was short of bytes.** The
phase timers first put a mature solve at 30.7 CPU-ms, against this document's
20 ms budget for 1,200 solves/s, and step 1 below reads as though a direct
sparse builder is the way to close that. It was not needed. Three things
accounted for a third of it, none of them the tree algorithm:

* the node array was a fresh `Vec` per solve that reserved 640 and doubled from
  there, while a mature subgame builds 2,039 nodes -- so every solve memcpied
  the array several times and first-touched megabytes of new pages;
* a `TNode` was 1,136 bytes, 688 of them a `State` that four places read, none
  of them in a hot loop. Terminal leaves now keep just their utility;
* `node_actions` cloned that same `State` once per reserve slot, and dedupd a
  few dozen action encodings through a freshly allocated `HashSet`.

Host cost is 20.4 CPU-ms/solve after those, and the remaining profile is flat:
the compact leaf row at 3.5 ms, the draw transitions at 2.9, serialization at
2.9, and nothing else above 1.9. There is no longer a single item worth a
rewrite, and `wait_frac` says the builders are only 64% busy -- the device is
the limit again on a generation-only stream. Add `--features prof` and call
`warchest.prof_dump()` to see the same table.

## What counts as success

The golden run is a real 30-minute `train.py` run on the two-RTX-3090 box, not
`gpu_bench`, not a fixed-tree microbenchmark, and not generation with training
disabled. The ReBeL phase uses the production settings:

- random drafts;
- depth 2 and 64 linear-CFR iterations;
- the production 384/64/64 network and changing real weights;
- optimizer batch size 1,024 and four optimizer samples per fresh solve;
- the current compact replay rows, mirror augmentation, target construction,
  snapshot schedule, 200,000-node safety cap, and horizon-payoff schedule.

The headline counter is:

```text
completed fresh ReBeL solves / ReBeL wall-clock seconds
```

The clock starts at the warm-to-ReBeL transition. It includes pipeline fill,
CPU tree construction, serialization, copies, CUDA inference and CFR, result
handling, replay insertion, optimizer work, weight publication, snapshots, and
the final partial interval. The process must obey the 30-minute deadline rather
than finish an arbitrarily long epoch after it.

Generation is only balanced if the trainer keeps up. At a train/generation
ratio of four, define optimizer credit as `optimizer_rows / 4`. The run must
report both raw generation rate and

```text
balanced solves = min(completed solves, optimizer credit)
```

with at most 1,024 owed optimizer samples--256 solves at a ratio of four--at
the end. A reported 1,200/s generator with a growing trainer backlog does not
meet the target. `docs/GPU_PERF_GOAL.md` pins the complete golden command.

The following are also hard requirements:

- zero games dropped because a GPU allocation or admission failed;
- a separate count of solver builds that really hit `Cfg::node_cap`;
- bounded CPU/GPU numerical comparisons, probability/index invariants, and a
  tight all-zero-network oracle in the FP32 baseline;
- no missing target rows, changed snapshot iterations, or silent precision
  mode;
- a `NOTES.md` for every training run.

Four counters must never again be collapsed into one word called "cap":

- `horizon_games`: completed games that reached `MAX_MAIN_PLAYS = 256`;
- `solver_node_caps`: attempted solves whose CPU tree hit 200,000 nodes and
  used the documented fallback;
- `oversize_routes`: valid jobs sent through the exact whale path;
- `dropped_games`: games abandoned because the system could not process a
  valid job. This must be zero.

## Corrected baseline and measurements

The old diagnosis mixed up two counters and one set of weights. `Data.cap_hits`
is incremented in `Game::finish` when a game reaches the 256-play horizon. It
does not count search trees hitting `Cfg::node_cap`. More importantly, the CUDA
services were started before warm training and did not receive the warm-started
weights before the first ReBeL batch. With `--rebel-games 1024`, that one stale
batch consumed the whole nominal run.

The difference is directly measurable on the same seed and horizon payoff:

| workload on two 3090s | solves/s | horizon games | network rows/solve | cells/solve | snapshot configs/solve |
|---|---:|---:|---:|---:|---:|
| freshly initialised seed-1 weights, 256 games | 130 | 254/256 | 1,707 | 137,087 | 146,642 |
| warm `gpu1200/snap_00`, 256 games | 297 | 86/256 | 1,015 | 17,239 | 19,283 |
| later strong checkpoint, historical best worker sweep | 573 | context only | 989 | -- | -- |

The first two rows used `seed=1000007`, horizon value `0.04`, random drafts,
depth 2, 64 iterations, real weights, 64 workers, four games per worker, and
the production-sized device pools. The fresh-weight replay took 250.5 seconds;
the warm-checkpoint replay took 75.4 seconds. The stale network keeps almost
every game alive to the horizon and preserves much broader beliefs, which
makes its sparse work roughly eight times larger even though its node count is
only 1.3 times larger.

After publishing the warm weights at the phase transition, the bounded real
trainer probe in `runs/arch_probe_published` completed three ReBeL batches:

| measurement | result |
|---|---:|
| completed and trained solves | 38,799 |
| ReBeL wall time | 248.5 s |
| end-to-end completed rate | **156 solves/s** |
| generator-side cumulative counter | 234--250 solves/s |
| target standard deviation | 0.291, 0.279, 0.249 |
| horizon-game fraction | 0.47, 0.48, 0.45 |
| nominal/final run time | 240.0 / 288.3 s |

The corrected rate is still far below the goal. It also shows why the current
counter is not the golden metric: `solves_per_s` is calculated immediately
after generation returns, before replay handling and the current training
pass. Large intervals are outside both logged `gen_s` and `train_s`, and the
epoch barrier overran the nominal budget by 48 seconds.

### GPU profile

The existing service is not compute-bound in the useful sense. A 33.4-second
Nsight Systems capture had 78% kernel-busy time but issued 874,367 kernels,
about 26,000 launches/s. Backpropagation used 20.8% of kernel time, SGEMMs
about 20.4%, head entry 11.1%, readout 8.7%, regret matching 7.9%, reach passes
about 12%, average-strategy work 4.0%, and belief work 4.1%. Another controlled
live-set capture reached 84.7% busy time. High utilization here means that the
device is continually launching small sparse kernels; it does not mean the
3090 is near its useful arithmetic or bandwidth limit.

On the intended warm-checkpoint replay the service averaged only 13,588 active
network rows and 6.4 admissions per batch. A production wave can present
32,000--64,000 contiguous rows without needing hundreds of resident game
threads. The profile transferred 11.97 GB host-to-device in 33.4 seconds, so
PCIe bandwidth was not the limiter either.

### Trainer profile

An isolated 1,024-row production batch on GPU 1 took about 173 ms with the
current `make_batch`; only about 14 ms was forward, backward, clipping, and
Adam. `np.unique` over the config key took about 105 ms and packed-row
expansion about 58 ms. The deduplication removed roughly 32% of config rows.

Computing the holding tower for every config row instead--with `inv` equal to
`arange`--is algebraically equivalent. Despite doing more GPU arithmetic, it
reduced the full step to 72 ms. With 320 generation workers active, the same
comparison was 228 ms with CPU deduplication versus 101 ms without it. The
1,200 target permits 213 ms per optimizer step; the 2,000 target permits 128
ms. Removing the sort is therefore both simpler and faster, and moving replay
sampling/expansion to the device provides the needed margin.

### Sparse-cell and CPU budgets

The frozen 1,000-root sample from `runs/pre_cuda_random/roots.bin` now reports
both dense and legal cells. At depth 2 and a real 200,000-node cap it found no
node-capped root:

| per root | median | p95 | p99 |
|---|---:|---:|---:|
| nodes | 661 | 4,522 | 18,072 |
| dense action cells | 5,166 | 71,773 | 158,800 |
| legal action cells | 2,355 | 33,392 | 98,062 |
| configs across nodes | 9,711 | 94,382 | 293,569 |

Only about half the dense `config x action` space is useful. The current
service nevertheless gives every dense cell regret, instantaneous regret,
current strategy, strategy sum, and average-strategy storage, plus a dense
successor entry. This is the dominant avoidable memory cost and is catastrophic
on the rare 100-million-cell roots.

On the warm-checkpoint replay, the summed `ADVANCE` timer was about 25.6 CPU-ms
per solve, including a 14.1 ms tree build, 1.1 ms serialization, 3.4 ms compact
public-feature work, and game/walk bookkeeping. At 1,200/s that is 30.7 CPU
seconds each second on a 36-core machine. It can reach 1,200, but leaves almost
no margin in its current form. Two thousand per second needs the compact sparse
builder to bring host work below roughly 15 ms/solve.

## End-to-end design

```text
 lightweight game actors
          |
          | needs a subgame
          v
 fixed CPU builder pool -- PackedJob + compact WalkTree
          |                       |
          | byte/work credits     `-- retained by the actor
          v
 cost buckets + oldest-first GPU dispatcher
          |
          +----> GPU 0 wave lanes ----+
          |                            |  sparse strategy, root values,
          `----> GPU 1 wave lanes ----+  carried-belief store
                         |             |
                         |             v
                         |       actor resumes and emits rows
                         |             |
                         |             v
                         `------ device replay ring on GPU 1
                                        |
                                  training CUDA Graph
                                        |
                                versioned weight banks
                                        `----> new GPU waves
```

This is a streaming system. A game is an ordinary Rust state machine, not an
OS thread. Ready actors run on a bounded work-stealing CPU pool. A submitted
solve parks the actor; a GPU completion puts it back on the ready queue. That
keeps all CPU cores useful without the current 320 blocked worker threads,
runnable-thread bursts, or hundreds of full trees retained merely to create
GPU occupancy.

Backpressure is expressed in bytes and work, never just job count. The host
tracks credits for packed tables, walk trees, pending result stores, replay
records, and each device lane. A job carries this work vector:

```text
(network rows, legal cells, reach slots, reverse nonzeros,
 table bytes, mutable bytes, carried-output bytes, levels)
```

The same vector selects a wave class, predicts service time, and prevents a
valid whale from exhausting host or device memory.

## CPU side: build once, retain only what play needs

The Rust rules and tree expansion stay on the CPU for the first implementation.
They are branchy, allocation-sensitive, and already verified; porting the
War Chest rule engine to CUDA would be a large independent correctness project.
The measured CPU budget says that work is not required for 1,200.

Use a fixed pool near the physical-core count. The exact split is tuned from
CPU utilization, but a sensible start is 30--34 tree builders plus a few cores
for game actors, result packing, logging, and the Python control thread. More
logical workers are allowed as actor records, not as concurrently runnable
tree builds.

The builder should produce two outputs directly:

- `PackedJob`: immutable sparse arrays ready to concatenate into a GPU wave;
- `WalkTree`: only the public tree information needed after the solve--node
  kind/player, public children, action metadata, config-support identity,
  sparse policy-row offsets, and leaf identifiers.

The full `Solver`, its build scratch, leaf-network buffers, reverse tables,
and duplicated node states are released as soon as those two objects exist.
Today `pending_sv` retains the whole solver while CUDA runs, which is why
increasing game concurrency can consume 125 GiB. The compact walk tree should
move or share the builder's support arrays rather than clone them.

For the 2,000/s stretch goal, profile this direct sparse builder again. The
host target is at most 20 ms/solve for 1,200 and 15 ms/solve for 2,000. Work in
this order if it misses:

1. construct sparse legal rows directly instead of building and then packing
   dense `legal`/`trans` arrays;
2. build `PackedJob` and `WalkTree` in the same traversal;
3. pool the large arenas by role and reserve from the measured work vector;
4. remove repeated public-row work and copies shown by the direct timers;
5. only then consider parallel expansion inside one whale tree.

A GPU rules/tree builder is a last gate, not part of the initial architecture.
It is justified only if the sparse direct builder cannot stay under the CPU
budget on the frozen workload tape.

## The sparse wave contract

The version-4 job's semantics are captured in git history and the frozen oracle
tapes; none of its runtime representation remains in production code. Version
5 is an immutable structure-of-arrays contract built around legal cells.

For each decision node and acting-player config, store a CSR row:

```text
legal_off[row .. row+1]
legal_action[cell]       local action id
legal_child[cell]        public child id, or derivable from the action
legal_trans[cell]        successor config in that child
```

Regret, current strategy, and strategy sum use this sparse cell index. Reverse
reach entries point to the same index. The game-side `NodePolicy` also becomes
sparse: sampling and the Bayes update already ignore illegal actions, so a
dense zero-filled row is unnecessary.

Before replacing the dense sentinel representation, capture an invariant and
golden public outputs over the frozen roots. Then replace it in place and check
that the sparse CPU solver reproduces those outputs byte-for-byte. Do not keep
dense and sparse solver implementations. If a rare rule needs a legal cell
with no successor, represent that case explicitly rather than infer it.

Use 32-bit wave-global offsets for clarity. Arrays whose values are proven
local and below 65,536--action ids, most config successors, and most leaf
config indices--may use 16 bits. A wave has a wide fallback when any job does
not fit; it must never truncate. Compression is accepted only when the tape
shows that reduced memory traffic beats unpack cost.

Several current fields disappear or change:

- `node_leaf` is redundant with `node_kind`;
- dense `legal_bits` and `trans` are replaced by legal-cell CSR;
- `inst` disappears;
- persistent `avg` disappears;
- per-solve table and arena offsets become one patched wave-global index
  space;
- hot tasks receive direct node/row maps, avoiding the current binary search
  from every flattened thread index back to a solve and then another search
  back to a node/config.

The mutable FP32 baseline consists of regret, current strategy, strategy sum,
reach, values, config/network embeddings, and bounded snapshot scratch. The
three cell arrays occupy about 12 bytes per legal cell, versus roughly 20 bytes
per dense cell today. On the frozen sample that is about a threefold reduction
before table compression; the reduction is larger on many broad-belief roots.

## GPU execution: bounded waves, not a resident tick

Each GPU owns two or three reusable wave lanes. A lane is a contiguous table
buffer, mutable arena, network-row buffer, result buffer, and a small pair of
snapshot staging buffers. Buffers are allocated by wave class, not by solve,
so there is no first-fit fragmentation and no high-water row span. The common
classes are learned from the production tape--for example small, medium,
large, and exclusive--and each is bounded in every work-vector dimension.

The dispatcher packs the oldest compatible jobs until a row/work target or a
short latency deadline is reached. A starting target of 32,000--64,000 network
rows per wave is large enough for the 384x64 GEMMs on a 3090. Jobs in one wave
share network shape, CFR rule, iteration count, snapshot list, and weight
version. Their arrays are concatenated once and local offsets are patched to
wave-global offsets.

Cost buckets prevent one giant tree from setting the shape or latency of a
normal wave. Oldest-first aging prevents starvation. The dispatcher routes to
the GPU with the lowest predicted finish time, not round robin. GPU 1's
prediction includes queued optimizer credit and its replay/training reserve.

The frozen tape can capture the fixed operation sequence as a CUDA Graph. Live
waves are different: profiling 809 waves found that cuBLAS/topology changes made
graph updates fail for most shapes, so recapture and instantiation cost 28.9 ms
per wave on average while queueing the useful work cost 40.3 ms. Direct stream
submission removes that construction cost and runs the identical fast FP32
kernels/GEMMs in the identical order. `WARCHEST_DIRECT=1` selects it while the
end-to-end crossover is measured. Do not create a graph variant for every
observed tree size.

This changes the launch economics. At 32 jobs/wave and 600 solves/s/card, a
card needs about 19 waves/s, not tens of thousands of host launches per second.
Whichever submission mode is used, each kernel sees one contiguous work list;
the wave itself, rather than graph capture, is the important launch-economics
change.

### CFR kernel schedule

Keep the CPU accumulation order encoded by the reverse gather tables. FP32 is
the reference path.

For each iteration:

1. form belief embeddings and run the two head GEMMs as large wave matrices;
2. read out leaf/config values;
3. sweep levels backward;
4. at an acting decision row, compute action values and the node value, then
   update discounted regret and the next current strategy in the same kernel;
5. sweep reaches forward and accumulate `reach * current` into strategy sum;
6. only at a requested snapshot, derive row normalization factors, propagate
   the average strategy, and stage leaf beliefs.

Fusing step 4 is safe: child values and the node value use the old current
strategy, while the newly regret-matched strategy is for the following reach
pass and iteration. The instantaneous-regret value never needs to survive the
row. Predictive CFR can use the just-computed delta before it is discarded.

Step 5 folds average-strategy accumulation into the level work whose reaches
are already hot. A naive CPU experiment that deferred average normalization
was 2% slower; that result does not argue for the current GPU kernel. The
proposal does not add a separate reread of every sum at every iteration. It
removes the separate average launch and only materialises row normalization at
the eight or so snapshots and at final output.

Use cuBLASLt or the existing cuBLAS path for the large `Wb` and `Wu` matrices,
with fused bias/activation epilogues where the exact LayerNorm order permits.
Do not replace a good GEMM with hand-written CUDA. The sparse reach,
backpropagation, readout, and row-normalization kernels are the custom part.

The production shape costs roughly 11 GFLOP per measured 1,000-row solve: the
per-iteration `Wb` and `Wu` products dominate. At 1,200/s that is about 13
TFLOP/s across two cards; at 2,000/s, about 22 TFLOP/s. Both are comfortably
below two 3090s' FP32 capacity. The target is physically plausible if waves
turn the small-matrix and launch-heavy workload into sustained work.

### Precision

FP32 CFR, reaches, reductions, outputs, and optimizer state are the baseline.
Network GEMMs use native, untiled cuBLAS SGEMM with its default fast math, and
the custom kernels are compiled with NVRTC's `--use_fast_math` umbrella flag.
`WARCHEST_GPU_PRECISE_MATH=1` is a diagnostic opt-out, not a production mode.
Do not pad matrices to a fixed oracle shape or select pedantic math merely to
make different batch shapes bit-identical. Associating an FP32 reduction
differently can change the last bits, and CFR can amplify that into a small
policy change. Correctness therefore comes from structural invariants, a tight
zero-network end-to-end oracle, and measured bounds against the CPU reference
rather than exact batch-composition identity.

Explicit TF32 was measured on the production tape after the wave engine was
working. It improved a long A/B from 577.9 to 587.7 solves/s, but changed one
eight-iteration policy probability by 0.094 after CFR amplification. The
zero-network oracle still agreed to about `6e-8`, so this was numerical
sensitivity rather than an indexing fault, but the trade was not worthwhile.
TF32 or BF16 may be revisited only if it produces a material end-to-end gain
and passes the target-statistics, `solvererr`, frozen-offline, and ladder gates.

It was revisited after the live tail was measurable. Explicit Ampere TF32 raised
an identical direct-launch tape from 581.1 to 597.4/s, but the 180-second warmed
live stream moved from 915.5 to 909.3/s before stop while producing more node
caps and exclusive routes. Two randomized-network bounds also failed, although
the zero-network and structural oracles passed. The experiment was reverted
because it did not make real generation faster, not because exact CPU floating
point agreement is required.

## Finish a solve without retaining its main GPU arena

TurboReBeL needs beliefs at the eventual exit leaf under each kept intermediate
average. The current service stores those beliefs for every possible leaf and
keeps the solve resident until the CPU walk names one leaf in trip 2. Snapshot
beliefs are one of the largest arenas on broad trees.

For normal waves, stream each snapshot's complete leaf-belief block to a host
`CarryStore` as soon as it is produced. Reuse a double-buffered device scratch
block at the next snapshot. Trip 1 returns a sparse final `StrategyStore`, root
values, the compact `WalkTree`, and the already complete `CarryStore`. The GPU
lane is then reusable. When the actor reaches a leaf, it selects the two spans
locally; there is no second device rendezvous.

The bandwidth is modest for the measured workload. The warm replay has about
19,000 snapshot configs/solve. Seven intermediate FP32 snapshots are roughly
0.53 MB/solve, or 0.64 GB/s at 1,200 solves/s across both cards. Sparse final
strategy and root values add well under 0.2 GB/s. The box exposes PCIe x16 and
the existing profile is far below link capacity.

The container has an 8 MiB locked-memory limit. Use a bounded pair of shared
pinned bounce buffers--for example one 3 MiB buffer per GPU, leaving headroom
for the runtime--and copy from them into pageable, byte-credited `CarryStore`
blocks. Raise the explicit lock limit before choosing larger buffers. Do not
try to pin one buffer per solve. Chunk a snapshot larger than the bounce
buffer.

An exclusive whale may have gigabytes of strategy/carry output. It still may
not be dropped. The first implementation can stream it to a separately
budgeted pageable store and admit no second whale until credits return. If
that cost is material on the production tape, add an exact rendezvous mode:
retain the exclusive whale's compact final state, return only policy rows along
the immediately executed CPU walk, gather the chosen leaf, and then release
it. This exceptional lane must not occupy or fragment the ordinary wave pools.

## Device-resident streaming replay and trainer

The 2-million-row replay is small relative to a 24 GiB card in its packed
form. Put the primary training ring on GPU 1:

- packed 223-byte public rows;
- per-row absolute config start and two support lengths;
- `uint8[15]` config counts;
- `uint8` player id;
- FP16 belief weights and targets;
- solve boundaries and age metadata.

Even the configured 96-million-config maximum is about 2.5 GiB with the public
rows and descriptors. Normal occupancy should be lower. Mirror it to pageable
host memory only when a dump or recovery path needs it.

Solve records arrive in small, ordered chunks and are copied asynchronously
into the ring. They are provisional at first. The frozen row format backfills
future-round auxiliary labels and the game result in `Game::finish`; a nonzero
`mc_mix` also changes the config targets there. Give every provisional segment
a game id and exclude it from sampling. At game end, copy the small timeline
patch, fill the auxiliary bytes, apply the optional outcome blend over the
segment's config spans, and atomically mark all of that game's solves
committed. A dropped or deadline-censored game invalidates its provisional
segments. Pending bytes participate in backpressure, and only committed solves
enter the headline and optimizer-credit counters. This preserves the existing
targets without retaining every game's large `Data` arenas on the host.
Fresh-only policy labels ride the same commit into a bounded transient queue
when the policy loss is enabled; with its production weight of zero, their
payload need not cross to the trainer.

A custom CUDA batch operator then:

1. samples row ids with the current uniform/recent mixture;
2. scans their two ragged support lengths into a fixed batch arena;
3. gathers packed rows and config records;
4. chooses mirror bits and applies the exact row/config/seat symmetry;
5. expands the public encoding from the frozen packed format;
6. forms one config feature per gathered config and sets `inv = arange`.

There is no CPU `np.unique`. Repeated configs are deliberately recomputed; the
measured 2.3--2.5x end-to-end step win already includes that extra arithmetic.
The operator must share the Rust row-layout constants and pass the existing
mirror/expansion self-check on a frozen sample.

Use fixed batch scratch and capture forward, loss, backward, clipping, and a
fused Adam step as a PyTorch/CUDA Graph where practical. Keep the optimizer
math and loss reductions FP32. The measured no-dedup Python path is already
101 ms while heavily contended, inside the 128 ms stretch budget; device replay
removes its remaining 58--82 ms host preparation and gives GPU 1 scheduling
headroom.

Track optimizer credit continuously. The trainer sleeps when it has no credit
and takes a high-priority GPU slice when debt approaches one batch. Solver
waves on GPU 1 use a lower-priority stream and sufficiently short kernels that
training can enter between graph nodes. The dispatcher sends proportionally
more solve work to GPU 0 while GPU 1 is repaying debt; it does not reserve an
entire card permanently for either role.

## Weight versions, not global drains

Each solve wave pins one immutable weight version. Each solve GPU owns two
weight banks. Publishing copies the roughly 3.4 MB flat network into the
inactive bank; new waves switch to it after an event, while old waves finish on
the previous bank. No live set drains and no in-flight solve changes weights.
Pack parameters into a preallocated device publication buffer after the
optimizer step--do not call `.cpu()` on every tensor. Copy GPU 1's bank
directly when peer access works; otherwise stage it asynchronously through the
same bounded pinned-bounce pool used for results.

Decouple publication cadence from the number of concurrent games. The
pre-CUDA trainer generated about 4,000 solves in its ordinary 48-game epoch and
published after roughly 16 optimizer steps. The 1,024-game CUDA batch silently
expanded that staleness to about 130,000 solves. Start with a fixed 16-step
publication interval, record the exact version on every replay row, and bound
staleness by one publication interval plus one wave. Changing this cadence is
an algorithm experiment, not a throughput tuning knob.

At the warm-to-ReBeL transition, synchronously install the warm snapshot in
both inactive banks before admitting the first ReBeL wave. The stale-weight bug
must have a regression test that reads back or hashes the version accepted by
the first job.

## Tails, caps, and deadlines

The distribution is heavy in cells, reach entries, and snapshot configs; node
count alone does not describe it. Schedule by the full work vector:

- normal jobs enter small/medium waves;
- large jobs get a lower-count large wave;
- a whale gets an exclusive lane and explicit host/device byte credits;
- a job too wide for the preferred compact indices uses the wide contract;
- a refused or failed solve is a hard error. There is no exact CPU fallback:
  it rebuilt multi-gigabyte arenas behind a global mutex, so the "last
  resort" serialized the whole run, and a path that only runs when something
  is already wrong is a path nothing measures.

Capacity refusal never abandons a game. If the 200,000-node build cap is truly
hit, preserve the existing uniform-policy fallback and count it as
`solver_node_caps`; that is an algorithmic safety rule, not a GPU allocation
failure. The 1,000-root frozen depth-2 sample had zero such hits, so a large
reported fraction is a monitoring bug until proven otherwise.

The run controller stops admitting new games and solves based on the hard wall
clock and measured p99 drain time. Work unfinished at the deadline is not
counted; partial end-of-run games can be discarded as time-censored work. It
does not wait another epoch. Snapshot writing is asynchronous and the final
checkpoint is taken from the last completed optimizer version inside the
budget.

## Performance budget

These are engineering budgets, not claimed speedups:

| component | measured now | 1,200/s budget | 2,000/s stretch |
|---|---:|---:|---:|
| intended warm-workload generation | 297 solves/s | 1,200 | 2,000 |
| host advance/build work | ~25.6 CPU-ms/solve | <=20 | <=15 |
| optimizer cadence | 4 samples/solve | 4.69 steps/s | 7.81 steps/s |
| 1,024-row trainer step | 101 ms no-dedup, contended | <=180 ms | <=110 ms |
| typical snapshot D2H | not streamed | <0.8 GB/s total | <1.3 GB/s total |
| capacity drops | nonzero historically | 0 | 0 |
| end-of-run training debt | unbounded by epoch | <=1 batch | <=1 batch |

A 32-solve wave must average at most about 53 ms on each card for 1,200 and
32 ms for 2,000, including its share of copies and result staging. Report wave
latency by class and weighted throughput, not only the fastest class.

Keep steady device allocation below about 20 GiB/card so CUDA, graph
executables, staging, and a rare wider wave have safety margin. GPU 1's budget
includes the replay ring and trainer. Keep host byte credits below a measured
safe ceiling--initially 80 GiB on the 125 GiB box--rather than relying on the
OOM killer.

## Validation and benchmark protocol

### Production workload tape

Before optimizing kernels, capture deterministic root/job tapes from a correct
CPU run at three weight ages: warm/early, middle, and late. Include ordinary,
p50, p95, p99, and the largest valid roots. A tape entry contains the state,
beliefs, exact weight version, packed job, carried roots, iteration/snapshot
metadata, CPU reference strategy, root values, and carried beliefs. Capture
representative replay batches too.

This avoids waiting minutes for one CPU game tail merely to compare a root and
makes every 2--5 minute performance experiment use the actual training
distribution. Report results weighted by the frequency and work contribution
of each tape class.

### Correctness gates

Every structural or kernel change passes, in order:

1. sparse CPU contract versus the frozen dense-CPU outputs on every tape root;
2. CPU contract round trip and internal index/transition invariants;
3. bounded per-phase FP32 CPU/GPU comparison on representative roots at 64
   iterations;
4. full strategy, root-value, and carried-belief comparison plus probability
   and shape invariants;
5. bounded wave-composition tests, including a whale beside small jobs;
6. all-zero-network identical-game trajectories for end-to-end scheduling
   changes;
7. target mean/std/quantiles and frozen offline learnability on a generated
   dump;
8. `solvererr` against the converged reference;
9. the final 30-minute training run and post-run ladder.

The ladder is a strength regression gate, not the way to choose between two
small performance edits. Use deterministic tapes and interleaved A/B runs for
those.

### Required live telemetry

At one-second and rolling-ten-second resolution, log:

- generated, inserted, trained-credit, and balanced solves/s;
- queue age and byte/work occupancy at every boundary;
- CPU build/pack/walk time and runnable builder count;
- waves/s, jobs/wave, rows/cells/reverse entries/wave, class latency, graph
  time, copy time, and useful GPU busy time per card;
- GPU clocks, power, temperature, and throttle reasons, so a rented-box limit
  is not mistaken for an architecture regression;
- trainer sample/expand/forward/backward/optimizer time and credit debt;
- weight version and age in solves/seconds;
- horizon games, real node caps, oversize routes, and drops;
- p50/p95/p99/max tree and output dimensions.

No phase may remain in an unlabelled gap between `gen_s` and `train_s`.

## Implementation sequence

### 0. Make the measurement true

- keep the warm-weight publication fix;
- rename the horizon metric and add real node-cap/oversize/drop counters;
- make the 30-minute controller interruptible and report balanced throughput;
- capture the production tapes and add complete phase timers;
- once the tapes are frozen, delete the v4 CUDA path before writing the v5
  executor.

Gate: a short trainer run uses the same first weight hash as `snap_00`, ends on
time, has zero unaccounted wall time, and replays deterministically.

### 1. Prove the sparse contract on CPU

- convert decision storage and `NodePolicy` to legal-cell CSR;
- build `PackedJob` and `WalkTree` directly;
- add narrow-index eligibility plus a wide fallback;
- validate against the frozen dense outputs; retain no dense fallback.

Gate: identical public outputs and all-zero trajectories, lower p50/p99 bytes,
and host work at or below 20 ms/solve on the box.

### 2. Build one FP32 wave executor

- one GPU, one common wave class, contiguous SoA buffers;
- direct task maps, fused backprop/regret update, fused reach/strategy-sum
  accumulation;
- CUDA Graph for admission, 64 iterations, value passes, and completion;
- host `CarryStore` spill through bounded pinned bounce buffers.

Gate: all GPU oracles pass and tape throughput is at least 600 solves/s on one
3090 for the weighted intended workload. This is the decisive architectural
test. If it misses, profile the graph before adding more concurrency or
precision modes.

Achieved on 2026-08-09. The first passing build reached 640.4 solves/s. Commit
`c40d246` reached 717.1 solves/s over a 32.0-second wall-clock interval on
one RTX 3090, including queue fill/drain, wave packing, transfers, graph work,
and result materialisation. The run used 64 production roots, three reusable
lanes, native FP32 SGEMM, fast NVRTC arithmetic, direct legal-cell value
indices, branch-grouped sparse tasks, and cooperative level sweeps.

### 3. Integrate the streaming runtime

- lightweight game actors and fixed builder pool;
- work-vector buckets, byte credits, two GPUs, dynamic routing, whale lane;
- compact `WalkTree`/strategy/carry completion and zero capacity drops;
- double-buffered weight versions with fixed publication cadence.

Gate: generation-only production tape and live self-play both exceed 1,400/s
with stable memory. The margin is intentional; training and tails still need
room.

Tape half achieved historically on 2026-08-09: commit `c40d246` reached 1,438.9
solves/s over a 32.1-second wall-clock interval across both RTX 3090s. The run
used six producers, three lanes per card, and `taskset -c 0-35` so GPU feeder
work stayed on one hardware thread per physical core. Exact whale routing and
cost isolation landed afterward. They make the heterogeneous frozen tape slower
but keep live memory bounded, so the current decision metric is aged live
self-play. Merging the two common live cost classes was measured there and
reduced the 180-second pre-stop rate from 915.5 to 892.4/s; fuller waves did not
repay their packing and transfer cost, and the change was reverted. Stable-memory
live self-play above 1,400/s remains required before this integration gate is
complete.

The first material live-tail improvement was removing the card-wide barrier for
exclusive whales. An aged trace showed that barrier leaving two of one card's
three streams idle. Commit `8a32c46` instead trims only the selected lane and
lets the other two continue. The 180-second warmed stream improved from 915.5
to 1,051.5/s before stop, and the real five-minute balanced trainer improved
from 624.5 to 699.7/s. Peak memory remained bounded at 21,473 MiB on the busier
24 GiB card while PyTorch trained concurrently on the other service card.

### 4. Replace the trainer boundary

- device replay ring and CUDA batch operator;
- no config deduplication;
- continuous optimizer credit, high-priority training, graph-captured step;
- hard deadline and asynchronous snapshots.

Gate: a five-minute real run sustains at least 1,200 balanced solves/s, carries
at most one batch of debt, and has zero drops or oracle regressions.

### 5. Golden run, then stretch

Done: `runs/gpu_golden8` is 1,315.4 balanced solves/s with a monotone ladder
behind it. What to do next is below.

## Where the next second comes from

Read this before starting any performance work, because the constraint has
moved and half the obvious ideas are already measured negatives.

**The device is the limit now, not the host.** On the 180-second mature
generation stream `wait_frac` is 0.45: builders spend nearly half their time
waiting on a card. Host cost is 20.4 CPU-ms per solve, down from 30.7, and the
remaining profile is flat -- compact leaf row 3.5 ms, draw transitions 2.9,
serialization 2.9, nothing else above 1.9. There is no single host item worth a
rewrite, and further host work currently shows up as idle builders rather than
throughput. That is why the cheap interning hash and the non-allocating draw
step measure as nothing: what they free cannot be used.

So, in order:

1. **Device work per solve.** About 1.45 ms of card time across both cards, with
   the cards 77% busy. The kernel mix is roughly readout 20%, backprop sweep 19%,
   reach sweep 15%, head entry 12%, belief sums 12%, GEMMs 10% -- so the network
   is *not* where the time goes; the sparse sweeps and the per-config gathers
   are. The readout was rewritten one config per lane for +2.6% because it spent
   a five-step shuffle reduction per config; `belief_sums` walks the same
   (row, config) pairs and deserves the same question. Do **not** retry FP16 for
   the `Z`/`G` config tables: a solve interns only ~200 distinct configs, so
   those tables are about a megabyte per wave and already L2-resident, which is
   why the earlier attempt measured -0.7%.

2. **The trainer's share of its card.** A high-priority stream took an optimizer
   step from ~240 ms back toward its uncontended 72-101, and that alone was
   worth 14% in the mature state -- but the trainer still owns a real fraction
   of GPU 1. Step 4 above (device replay ring, graph-captured step) is the
   remaining piece. `WARCHEST_WAVE_LANES` now takes a per-device list; 12/6 was
   far too extreme, and something like 10/8 is untested.

3. **The host, only if it becomes binding again.** In order: the compact leaf
   row, the draw transitions, serialization, and flattening `TNode`'s fourteen
   per-node vectors into solver-level arenas.

### Measured negatives — do not repeat these

* merging the wave cost classes: +14% on the frozen tape, 5% *worse* live;
* a deeper in-flight window (64 or 80 per builder): better on the
  generation-only stream, and a collapse to 425 solves/s with the trainer
  present, because it doubles the trees retained on the host;
* twelve lanes per card: best on the generation-only stream, out of card memory
  within ten minutes once the trainer shares it;
* cooperative sweep grid sizes (1-8 blocks per SM): flat;
* mimalloc as a Rust `#[global_allocator]`: exhausts the static TLS block and
  torch then cannot load libgomp. jemalloc via `LD_PRELOAD` is worth ~4%;
* keeping the training loss on the device without also prefetching the next
  batch: the step is GPU-bound, so removing one sync alone changes nothing.

The first three of those all share a shape: they were measured on a benchmark
with no trainer and a horizon too short to feel retained memory. Measure with
`tools/v5_steady.sh`, and compare at equal cumulative solves.

## Designs not to pursue first

- More generation threads or larger game batches. They increase retained
  trees, CPU contention, staleness, and tail barriers; they do not regularize
  GPU work.
- A faster `gpu_bench` as the stop condition. It is a kernel tool, not the
  training objective.
- Static round-robin device routing. GPU 1 has replay and optimizer work, and
  job costs vary by orders of magnitude.
- Dropping or truncating an oversized valid job. That biases training toward
  easy positions.
- A single monolithic persistent kernel containing the neural network. CUDA
  Graphs plus good library GEMMs are simpler and preserve the verified phase
  boundaries. Consider a cooperative sparse-sweep kernel only if a graph
  profile still makes launch/device scheduling dominant.
- Porting the rules engine to CUDA before measuring the sparse CPU builder.
- TF32 as a prerequisite. The current FP32 arithmetic budget is sufficient;
  precision is a later, separately gated optimization.

The central bet is deliberately testable: one contiguous, sparse, FP32 wave
must sustain 600 weighted solves/s on one 3090. If it does, the rest of the
architecture removes the host, trainer, tail, and versioning reasons that two
such cards would fail to turn 1,200 raw solves/s into 1,200 real training
solves/s. If it does not, the tape and graph profile will say which physical
budget was wrong before the codebase commits to the full rewrite.
