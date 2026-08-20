# Student of Games, and where the device port stands

The engine follows Student of Games (`papers/SoG_2112.03178.pdf`), not ReBeL.
Four things the paper specifies; three are in and verified, the fourth is half
done.

## What the paper asks for, and what answers it

**A counterfactual value-and-policy network**, `f(β) = (v, p)`. The second
readout has the same shape as the first: value is a config vector dotted with a
situation vector, `v(c) = <f(c), h>`, and policy is `logit(c, a) = <f_p(c),
e(a)>`. `f_p` is a third head off the config encoder that already produces `f`
and `g`; `e(a)` describes one action — its kind, the coin slot it spends, the
three squares it names — against the board it is played on. Both land on the
`(config, action)` cells the tree already indexes for its strategy, so the
policy needs no table of its own and costs one dot product per cell.

An action's private content is *whether it is legal*, and the tree carries that
as the legal cells, which is what lets one public description serve every
config at a node. `train/test_parity.py::policy_parity` is the only thing that
pins `cfg_p` and the action encoder's place in the weight blob.

**Expansion by `π_select = ½·π_PUCT + ½·π_CFR`.** PUCT is a maximisation, so
its half is a point mass on the argmax and sampling the mixture is a coin flip.
Three arenas laid out like `cur` carry it: `prior`, `visits` (incremented as a
trajectory passes, which is also the paper's virtual loss), and `qval`, the
action value `backprop` already forms. Q is divided by the opponent's reach
mass at the node — without it a node behind an unlikely opponent line looks
worthless beside its own siblings instead of being compared with them.

The prior is filled at the start of an expansion phase, not inside `grow`: a
node is expanded before the batch carrying its board vector has run. Only
expanded nodes ever need one, because a leaf has no action list and a
trajectory stops there.

**The regret update phase**: "simultaneous updates, regret-matching+, and
linearly-weighted policy averaging". `Cfr::SOG` is `alpha = inf, beta = -inf,
gamma = 1`. The gamma needs care — a sum decayed by `(t/(t+1))^gamma` weights
iterate `j` by `(j+1)^gamma`, so *linear* is 1, not the 2 `Cfr::PLUS` carries.
`step` traverses both players against one reach profile.

**The CFR loop on the device.** Half done. See below.

## The device

The wall was never arithmetic. Throughput pinned at ~380k network rows/s
whatever the thread count, with the CPUs at 13% and the cards at 20%: ~3 KB
crossed the bus per join row per iteration and 87% of it was data the card had
just produced. So a solve's board vectors, `f`, `g` and belief index stay with
the backend, a round shards by *solve* so every call of one reaches the backend
holding its state, and `Call::Leaf` is a whole CFR iteration — beliefs in,
counterfactual values out. The pooled block and the head never leave.

`farm::leaf` is the CPU reference and the oracle the kernels answer to.
`a_gated_solve_matches_an_ungated_one_exactly` holds the batched and unbatched
paths to byte equality on packed rows, beliefs and targets.

**The CFR sweeps.** `contract.rs` describes the tree as flat arrays with the
reach transition transposed from a scatter into a gather; both sweeps reproduce
the solver bit for bit, and a whole solve driven from the description reaches a
byte-identical strategy. `Contract::extend` keeps the description current for 49
cpu-ms per three solves where a rebuild cost 1651 — the tax that made the port
not worth doing at all. `k_reach_sweep` and `k_backprop_sweep` transcribe the
two, one block per (node, player) with threads over that node's configs, so
neither needs a task list.

`Device::keep_tree` and `Device::sweep` drive them: reach forward from level
one, backpropagation backward, one launch a level. The sweeps are compiled but
not yet wired into a solve.

**Where it stands, measured.** The fused leaf pass agrees with the CPU network
to 2.6e-6 on one card and on two, with batch invariance holding. Row throughput
went from 380k/s before the fusion to **1.1M/s** after — but it is still a
wall: 36, 72 and 144 threads all land there, at a constant **0.9 µs a row**,
while the cards sit at **14–27%** utilisation and the host waits inside
`Backend::run` for 91% of wall clock.

0.9 µs a row is ~330 GFLOP/s against the ~71 TFLOP/s the two cards have. Half a
percent of peak. Whatever costs that time, it is not arithmetic.

Timed inside the leaf pass, over a 120 s probe on two cards:

| | ms | share |
|---|---:|---:|
| host marshalling | 13,851 | 24% |
| uploads | 3,505 | 6% |
| launches | 35,696 | **61%** |
| the one download | 5,523 | 9% |

The guess above — pageable uploads serialising the stream — was wrong. Uploads
are 6%. **Launches are 61%, and they cost ~52 µs a call**, against the 5–10 µs
a launch normally takes.

The cause is structural rather than a tuning error. `g` and `f` are resident
*per solve*, so the pooling and the readout cannot be one launch over the round
— they are one launch per solve, twice, and a round holds thirty-odd solves.
Small kernels, thousands a second, and the cards idle between them.

The fix is the layout the architecture at `f5f4c05^` used and this one dropped:
one card-wide arena with each solve occupying a slice of it, so a stage is a
single launch over the whole round with per-query base offsets, rather than one
launch a solve. That is the next thing to build, and it is the same shape as
every other finding here — the traffic and the launches both come from treating
a solve as the unit when the round is.

At the frozen budget this is **1.5 solves/s**, against 1.4 before. The fusion's
2.9x in rows/s was spent exactly, and only, on the doubling of rows a solve that
simultaneous updates brought.

**But the sweeps cannot move on their own, and this is the thing to settle
first.** `sample_leaf` — the expansion phase — runs on the host and reads
exactly the arenas the sweeps write: `cur`, `sum_strat`, `qval`, `visits`,
`prior`, `reach`.

Measured, those arenas are **4.9, 9.7 and 33.1 MB** over three real solves — a
mean of 16 MB a round trip, not the 2 MB a rough estimate suggested. Over 64
iterations that is **1 GB a solve**, and at 150 solves/s **153 GB/s**, against
roughly 50 GB/s across two PCIe 4.0 x16 links. Three times over the bus, before
any of the network's own traffic. Making the arenas resident and leaving
expansion on the host is not slow, it is impossible.

Walking the trajectory from the host instead of bulk-copying does not rescue
it: a trajectory is data-dependent and sequential, so it becomes ~160 tiny
round trips an iteration, and latency replaces bandwidth as the wall.

So either the expansion phase moves to the device with the sweeps — which is
what the architecture at `f5f4c05^` did, trajectories and PUCT statistics on the
card, nothing returning per iteration — or the sweeps stay on the host at the
~146 solves/s ceiling they impose. There is no version where they move alone.

A sweep must hand back `qval` as well as the values. The expansion phase reads
it as PUCT's Q, and a sweep that computes each action value and drops it leaves
selection blind: the numbers stay right and the search grows a different tree.
Per-iteration comparisons cannot see this, because each one starts from a tree
the walk advanced. Only driving a whole solve from the description shows it.

## Numbers to size against

Measured at `nodes=8192, expand=8, iters=64` over real roots.

* An expansion adds ~17 nodes, so tree size is set by `iters × expand` and the
  `nodes` budget barely binds; growth finishes near iteration 38 of 64.
* Per solve: 10,175 trunk rows, 422k join row-queries, 41.9M readout configs =
  312 GFLOP. **150 solves/s is 47 TFLOP/s** against ~71 of FP32 across two
  3090s, so it needs the tensor cores.
* The network is 93% of a solve — but the CFR sweeps are 73% of what is *left*
  once it moves, and that residual caps the host near **146 solves/s** until
  they move too. Both ends are tight at 150.
* The flat gather is *slower* than the tree walk on a CPU (0.7x). It buys
  parallelism with work and there is nobody on a host to spend it, so both
  implementations stay.
* **A solve's cost varies 26x.** Strategy cells per solve run 67k to 1.73M over
  eight real roots — 8 to 227 cells a node — and the single largest solve of the
  eight carries 40% of all the cells. `config_cap` bounds the *root* belief;
  interior supports still multiply through every draw, so a tree that reaches
  past a round boundary explodes where a shallow one does not.

  Two consequences for the device. Memory: 35 MB of cell arenas for the largest
  solve, ~390 MB for thirty-six at the mean but 1.2 GB if they are all large,
  on a 24 GB card. And load: the sweeps are O(cells), so a round holding one
  large solve and thirty-five small ones is as slow as its largest member. The
  architecture at `f5f4c05^` learned the same lesson twice — reserved lanes for
  oversized work, and returning outsized buffers rather than retaining them.
* **A solve's cost cannot be predicted from its root.** The obvious way to feed
  reserved lanes is to route by root belief size. It does not work: three solves
  with an identical root support of 124 came to 203k, 1.32M and 1.73M cells, an
  8.5x spread, while the *largest* root of the eight (278) produced middling
  trees. Supports do grow away from the root — up to 10.7x, so `config_cap`
  bounding the root bounds nothing else — but where they grow depends on which
  lines the search chooses to follow, which is not known when the solve starts.
  A scheduler therefore has to react to cost as it accrues, not sort by it up
  front.
